import type { VolumeMetadata, VoxelTypedArray, SliceData, ViewAxis } from '../types.js';

/**
 * Core container for 3D volume data.
 * Row-major layout: index = x + y*nx + z*nx*ny
 */
export class VolumeData {
    readonly isStreaming = false;
    readonly dimensions: [number, number, number];
    readonly dataType: string;
    readonly spacing: [number, number, number];
    readonly metadata: VolumeMetadata;
    readonly data: VoxelTypedArray;
    readonly min: number;
    readonly max: number;

    constructor(data: VoxelTypedArray, metadata: VolumeMetadata) {
        this.dimensions = metadata.dimensions;
        this.dataType = metadata.dataType;
        this.spacing = metadata.spacing || [1.0, 1.0, 1.0];
        this.metadata = metadata;
        this.data = data;

        if (metadata.min !== undefined && metadata.max !== undefined) {
            this.min = metadata.min;
            this.max = metadata.max;
        } else {
            const [min, max] = this.calculateMinMax();
            this.min = min;
            this.max = max;
        }
    }

    /** Create VolumeData from a raw ArrayBuffer + metadata */
    static fromArrayBuffer(buffer: ArrayBuffer, metadata: VolumeMetadata): VolumeData {
        const [nx, ny, nz] = metadata.dimensions;
        const expectedSize = nx * ny * nz;

        let data: VoxelTypedArray;
        switch (metadata.dataType) {
            case 'uint8':
                if (buffer.byteLength < expectedSize) {
                    throw new Error(`Buffer size mismatch: expected ${expectedSize} bytes, got ${buffer.byteLength}`);
                }
                data = new Uint8Array(buffer, 0, expectedSize);
                break;
            case 'uint16':
                if (buffer.byteLength < expectedSize * 2) {
                    throw new Error(`Buffer size mismatch: expected ${expectedSize * 2} bytes, got ${buffer.byteLength}`);
                }
                data = new Uint16Array(buffer, 0, expectedSize);
                break;
            case 'float32':
                if (buffer.byteLength < expectedSize * 4) {
                    throw new Error(`Buffer size mismatch: expected ${expectedSize * 4} bytes, got ${buffer.byteLength}`);
                }
                data = new Float32Array(buffer, 0, expectedSize);
                break;
            default:
                throw new Error(`Unsupported data type: ${metadata.dataType}`);
        }

        return new VolumeData(data, metadata);
    }

    /** Extract a 2D slice from the volume */
    getSlice(axis: ViewAxis, index: number): SliceData {
        const [nx, ny, nz] = this.dimensions;

        switch (axis) {
            case 'xy': {
                if (index < 0 || index >= nz) {
                    throw new Error(`Slice index ${index} out of bounds [0, ${nz})`);
                }
                const offset = index * nx * ny;
                return {
                    data: this.data.slice(offset, offset + nx * ny),
                    width: nx,
                    height: ny,
                };
            }
            case 'xz': {
                if (index < 0 || index >= ny) {
                    throw new Error(`Slice index ${index} out of bounds [0, ${ny})`);
                }
                const sliceData = new (this.data.constructor as new (len: number) => VoxelTypedArray)(nx * nz);
                for (let z = 0; z < nz; z++) {
                    for (let x = 0; x < nx; x++) {
                        sliceData[x + z * nx] = this.data[x + index * nx + z * nx * ny];
                    }
                }
                return { data: sliceData, width: nx, height: nz };
            }
            case 'yz': {
                if (index < 0 || index >= nx) {
                    throw new Error(`Slice index ${index} out of bounds [0, ${nx})`);
                }
                const sliceData = new (this.data.constructor as new (len: number) => VoxelTypedArray)(ny * nz);
                for (let z = 0; z < nz; z++) {
                    for (let y = 0; y < ny; y++) {
                        sliceData[y + z * ny] = this.data[index + y * nx + z * nx * ny];
                    }
                }
                return { data: sliceData, width: ny, height: nz };
            }
        }
    }

    /** Get the value at a specific 3D coordinate */
    getValue(x: number, y: number, z: number): number | null {
        const [nx, ny, nz] = this.dimensions;
        if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) {
            return null;
        }
        return this.data[x + y * nx + z * nx * ny];
    }

    /** Get summary info about the volume */
    getInfo(): {
        dimensions: [number, number, number];
        dataType: string;
        spacing: [number, number, number];
        range: [number, number];
        totalVoxels: number;
        memorySizeMB: string;
    } {
        return {
            dimensions: this.dimensions,
            dataType: this.dataType,
            spacing: this.spacing,
            range: [this.min, this.max],
            totalVoxels: this.data.length,
            memorySizeMB: (this.data.byteLength / (1024 * 1024)).toFixed(2),
        };
    }

    /** Check if this is a single-slice volume (2D image) */
    isSingleSlice(): boolean {
        return this.dimensions[2] === 1;
    }

    /** Check if this is an RGB volume */
    isRGB(): boolean {
        return this.metadata.isRGB === true;
    }

    /** Get channel label for RGB volumes */
    getChannelLabel(zIndex: number): string | null {
        if (this.metadata.isRGB && this.dimensions[2] === 3) {
            return ['Red', 'Green', 'Blue'][zIndex] || null;
        }
        return null;
    }

    /** Async slice access — wraps sync getSlice in a resolved Promise */
    getSliceAsync(axis: ViewAxis, index: number): Promise<SliceData> {
        return Promise.resolve(this.getSlice(axis, index));
    }

    /** Returns the volume to use for MIP rendering (self for in-memory volumes) */
    getMIPVolume(): VolumeData {
        return this;
    }

    /** Trigger prefetch — no-op for in-memory volumes */
    prefetch(_axis: ViewAxis, _index: number): void {
        // no-op
    }

    /** Cleanup — no-op for in-memory volumes */
    dispose(): void {
        // no-op
    }

    /**
     * Check if 3D volume can be downsampled (dimensions > 1 on all axes).
     */
    canEnhance3D(): boolean {
        const [nx, ny, nz] = this.dimensions;
        return nx > 1 && ny > 1 && nz > 1;
    }

    /**
     * Create a downsampled 3D volume from in-memory data.
     * @param scale Downsample factor (2 = half resolution, 4 = quarter)
     */
    async createDownsampledVolume(scale: number, onProgress?: (progress: number) => void): Promise<VolumeData | null> {
        const [nx, ny, nz] = this.dimensions;
        const dstNx = Math.ceil(nx / scale);
        const dstNy = Math.ceil(ny / scale);
        const dstNz = Math.ceil(nz / scale);

        console.log(`Downsampling 3D: Creating ${dstNx}×${dstNy}×${dstNz} volume (scale=${scale})`);

        const enhancedData = new Float32Array(dstNx * dstNy * dstNz);

        for (let dz = 0; dz < dstNz; dz++) {
            const srcZ = dz * scale;
            if (srcZ >= nz) break;
            const srcZOffset = srcZ * nx * ny;
            const dstZOffset = dz * dstNx * dstNy;

            for (let dy = 0; dy < dstNy; dy++) {
                const srcY = dy * scale;
                if (srcY >= ny) break;
                const srcYOffset = srcZOffset + srcY * nx;
                const dstYOffset = dstZOffset + dy * dstNx;

                for (let dx = 0; dx < dstNx; dx++) {
                    const srcX = dx * scale;
                    if (srcX >= nx) break;
                    enhancedData[dstYOffset + dx] = this.data[srcYOffset + srcX];
                }
            }

            if (onProgress && dz % Math.max(1, Math.floor(dstNz / 20)) === 0) {
                onProgress(Math.floor((dz + 1) / dstNz * 100));
            }
        }

        if (onProgress) {
            onProgress(100);
        }

        return new VolumeData(
            enhancedData,
            {
                ...this.metadata,
                dimensions: [dstNx, dstNy, dstNz],
                dataType: 'float32'
            }
        );
    }

    private calculateMinMax(): [number, number] {
        let min = Infinity;
        let max = -Infinity;
        for (let i = 0; i < this.data.length; i++) {
            const v = this.data[i];
            if (v < min) min = v;
            if (v > max) max = v;
        }
        return [min, max];
    }
}
