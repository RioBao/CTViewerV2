import type { VolumeMetadata, VoxelDataType, VoxelTypedArray, SliceData, ViewAxis } from '../types.js';
import { VolumeData } from './VolumeData.js';

const LRU_CACHE_SIZE = 100;
const PREFETCH_RADIUS = 10;
const MAX_CONCURRENT_PREFETCH = 4;
const SLAB_BUDGET = 50 * 1024 * 1024; // ~50MB per slab read

/** Bytes per element for a given voxel data type */
function bytesForType(dt: VoxelDataType): number {
    switch (dt) {
        case 'uint8': return 1;
        case 'uint16': return 2;
        case 'float32': return 4;
    }
}

/** Create a typed array view over an ArrayBuffer at a byte offset (zero-copy) */
function typedArrayView(buf: ArrayBuffer, dt: VoxelDataType, byteOffset: number, length: number): VoxelTypedArray {
    switch (dt) {
        case 'uint8': return new Uint8Array(buf, byteOffset, length);
        case 'uint16': return new Uint16Array(buf, byteOffset, length);
        case 'float32': return new Float32Array(buf, byteOffset, length);
    }
}

/** Create a typed array view over an entire ArrayBuffer */
function bufferToTypedArray(buf: ArrayBuffer, dt: VoxelDataType, count: number): VoxelTypedArray {
    return typedArrayView(buf, dt, 0, count);
}

/**
 * Streaming volume data for files too large to fit in memory.
 * Holds a File reference and reads slices on demand.
 * Uses a 4x-downsampled VolumeData for MIP rendering and fallback slices.
 *
 * Architecture mirrors the old viewer's StreamingVolumeData:
 * - XY slices: serialized I/O queue with concurrent prefetch (cap=4)
 * - XZ/YZ slices: independent async loads with cancellation, center-out slabs,
 *   zero-copy typed array views for row extraction, progressive callbacks per slab
 */
export class StreamingVolumeData {
    readonly isStreaming = true;
    readonly dimensions: [number, number, number];
    readonly dataType: VoxelDataType;
    readonly spacing: [number, number, number];
    readonly metadata: VolumeMetadata;
    readonly min: number;
    readonly max: number;

    /** Callback fired when an async slice is ready â€” ViewerApp should re-render */
    onSliceReady: ((axis: ViewAxis, index: number) => void) | null = null;

    private file: File;
    private lowRes: VolumeData;
    private bpe: number; // bytes per element
    private sliceBytes: number; // bytes per XY slice in file

    // XY slice LRU cache (Map insertion order for LRU)
    private xyCache = new Map<number, SliceData>();

    // Serialized I/O queue for XY reads only
    private xyQueue: Promise<void> = Promise.resolve();

    // Prefetch tracking â€” limits concurrent reads
    private prefetchInProgress = new Set<number>();

    // Separate XZ/YZ caches (one slot each, like old viewer)
    private currentXZSlice: SliceData | null = null;
    private currentXZIndex = -1;
    private xzLoadInProgress = false;

    private currentYZSlice: SliceData | null = null;
    private currentYZIndex = -1;
    private yzLoadInProgress = false;

    constructor(file: File, metadata: VolumeMetadata, lowRes: VolumeData) {
        this.file = file;
        this.metadata = metadata;
        this.dimensions = metadata.dimensions;
        this.dataType = metadata.dataType;
        this.spacing = metadata.spacing;
        this.min = lowRes.min;
        this.max = lowRes.max;
        this.lowRes = lowRes;

        this.bpe = bytesForType(metadata.dataType);
        const [nx, ny] = metadata.dimensions;
        this.sliceBytes = nx * ny * this.bpe;
    }

    /**
     * Synchronous slice access â€” returns cached full-res or upscaled low-res fallback.
     * Always returns immediately.
     */
    getSlice(axis: ViewAxis, index: number): SliceData {
        if (axis === 'xy') {
            // Check XY LRU cache
            const cached = this.xyCache.get(index);
            if (cached) {
                // Move to end for LRU freshness
                this.xyCache.delete(index);
                this.xyCache.set(index, cached);
                return cached;
            }
        } else if (axis === 'xz') {
            if (this.currentXZSlice && this.currentXZIndex === index) {
                return this.currentXZSlice;
            }
        } else {
            if (this.currentYZSlice && this.currentYZIndex === index) {
                return this.currentYZSlice;
            }
        }

        // Return upscaled low-res fallback
        return this.getLowResSlice(axis, index);
    }

    /**
     * Async slice access â€” reads from file and caches. Returns full-res data.
     * For XY: single contiguous read via serialized queue.
     * For XZ/YZ: independent slab-based progressive loading with cancellation.
     */
    getSliceAsync(axis: ViewAxis, index: number): Promise<SliceData> {
        if (axis === 'xy') {
            const cached = this.xyCache.get(index);
            if (cached) {
                this.xyCache.delete(index);
                this.xyCache.set(index, cached);
                return Promise.resolve(cached);
            }
            return this.readXYSlice(index);
        } else if (axis === 'xz') {
            if (this.currentXZSlice && this.currentXZIndex === index && !this.currentXZSlice.isLowRes) {
                return Promise.resolve(this.currentXZSlice);
            }
            this.loadXZSliceAsync(index);
            // Return immediately with whatever we have
            return Promise.resolve(this.getSlice(axis, index));
        } else {
            if (this.currentYZSlice && this.currentYZIndex === index && !this.currentYZSlice.isLowRes) {
                return Promise.resolve(this.currentYZSlice);
            }
            this.loadYZSliceAsync(index);
            return Promise.resolve(this.getSlice(axis, index));
        }
    }

    /** Returns the 4x-downsampled volume for MIP rendering */
    getMIPVolume(): VolumeData {
        return this.lowRes;
    }

    /** Trigger prefetch of nearby XY slices */
    prefetch(axis: ViewAxis, index: number): void {
        if (axis !== 'xy') return; // Only prefetch XY (cheap contiguous reads)

        const nz = this.dimensions[2];
        const toFetch: number[] = [];

        // Build ordered list nearest-first
        for (let d = 1; d <= PREFETCH_RADIUS; d++) {
            const forward = index + d;
            if (forward < nz && !this.xyCache.has(forward) && !this.prefetchInProgress.has(forward)) {
                toFetch.push(forward);
            }
            const backward = index - d;
            if (backward >= 0 && !this.xyCache.has(backward) && !this.prefetchInProgress.has(backward)) {
                toFetch.push(backward);
            }
        }

        // Only launch up to available concurrent slots
        const available = MAX_CONCURRENT_PREFETCH - this.prefetchInProgress.size;
        const batch = toFetch.slice(0, Math.max(0, available));

        for (const z of batch) {
            this.prefetchInProgress.add(z);
            this.readXYSlice(z)
                .then(() => this.prefetchInProgress.delete(z))
                .catch(() => this.prefetchInProgress.delete(z));
        }
    }

    /** Get the value at a specific 3D coordinate (from low-res approximation) */
    getValue(x: number, y: number, z: number): number | null {
        const [nx, ny, nz] = this.dimensions;
        if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) return null;

        const [lx, ly, lz] = this.lowRes.dimensions;
        const scale = this.lowRes.dimensions[0] > 0 ? nx / lx : 4;
        const lxi = Math.min(lx - 1, Math.floor(x / scale));
        const lyi = Math.min(ly - 1, Math.floor(y / scale));
        const lzi = Math.min(lz - 1, Math.floor(z / scale));
        return this.lowRes.getValue(lxi, lyi, lzi);
    }

    /** Volume info summary */
    getInfo(): {
        dimensions: [number, number, number];
        dataType: string;
        spacing: [number, number, number];
        range: [number, number];
        totalVoxels: number;
        memorySizeMB: string;
    } {
        const totalVoxels = this.dimensions[0] * this.dimensions[1] * this.dimensions[2];
        return {
            dimensions: this.dimensions,
            dataType: this.dataType,
            spacing: this.spacing,
            range: [this.min, this.max],
            totalVoxels,
            memorySizeMB: ((totalVoxels * this.bpe) / (1024 * 1024)).toFixed(2),
        };
    }

    isSingleSlice(): boolean {
        return this.dimensions[2] === 1;
    }

    isRGB(): boolean {
        return this.metadata.isRGB === true;
    }

    getChannelLabel(zIndex: number): string | null {
        if (this.metadata.isRGB && this.dimensions[2] === 3) {
            return ['Red', 'Green', 'Blue'][zIndex] || null;
        }
        return null;
    }

    /** Cleanup file reference and caches */
    dispose(): void {
        this.xyCache.clear();
        this.prefetchInProgress.clear();
        this.currentXZSlice = null;
        this.currentYZSlice = null;
        this.currentXZIndex = -1;
        this.currentYZIndex = -1;
        this.onSliceReady = null;
        // Release large retained objects so GC can reclaim them even if
        // pending promise chains still reference `this`.
        (this as unknown as { file: File | null }).file = null;
        (this as unknown as { lowRes: VolumeData | null }).lowRes = null;
    }

    // ========================================================================
    // XY slice reading (serialized queue, like old viewer's readBlob chain)
    // ========================================================================

    /** Read a single XY slice â€” contiguous in file, fast (~4MB, <50ms) */
    private readXYSlice(index: number): Promise<SliceData> {
        // Already cached?
        const cached = this.xyCache.get(index);
        if (cached) {
            this.xyCache.delete(index);
            this.xyCache.set(index, cached);
            return Promise.resolve(cached);
        }

        const [nx, ny] = this.dimensions;
        const offset = index * this.sliceBytes;

        // Enqueue serialized read (prevents I/O thrashing)
        const readPromise = new Promise<SliceData>((resolve) => {
            this.xyQueue = this.xyQueue.then(async () => {
                // Re-check cache (may have been loaded by a concurrent path)
                const c = this.xyCache.get(index);
                if (c) {
                    resolve(c);
                    return;
                }

                const buf = await this.file.slice(offset, offset + this.sliceBytes).arrayBuffer();
                const data = bufferToTypedArray(buf, this.dataType, nx * ny);
                const slice: SliceData = { data, width: nx, height: ny };

                this.putXYCache(index, slice);
                this.onSliceReady?.('xy', index);
                resolve(slice);
            });
        });

        return readPromise;
    }

    // ========================================================================
    // XZ slice loading â€” independent async, cancellation, center-out slabs
    // ========================================================================

    /**
     * Async load high-res XZ slice using slab-based reads.
     * Reads contiguous blocks of XY slices as single blobs (~50MB each),
     * extracts row y using typed array views (zero-copy), center-out order,
     * fires onSliceReady after each slab for progressive display.
     * Cancels if the requested index changes mid-load.
     */
    private async loadXZSliceAsync(y: number): Promise<void> {
        // If already loading this exact index, don't restart
        if (this.xzLoadInProgress && this.currentXZIndex === y) return;

        const [nx, ny, nz] = this.dimensions;
        if (y < 0 || y >= ny) return;

        this.xzLoadInProgress = true;
        this.currentXZIndex = y;
        this.currentXZSlice = null;

        try {
            // Initialize with upscaled low-res data
            const sliceData = new Float32Array(nx * nz);
            const [lnx, lny, lnz] = this.lowRes.dimensions;
            const lowData = this.lowRes.data;
            const scale = Math.round(nx / lnx);
            const ly = Math.min(Math.floor(y / scale), lny - 1);
            for (let z = 0; z < nz; z++) {
                const lz = Math.min(Math.floor(z / scale), lnz - 1);
                for (let x = 0; x < nx; x++) {
                    const lx = Math.min(Math.floor(x / scale), lnx - 1);
                    sliceData[x + z * nx] = lowData[lx + ly * lnx + lz * lnx * lny];
                }
            }

            // Slab sizing
            const slabSlices = Math.max(1, Math.floor(SLAB_BUDGET / this.sliceBytes));
            const totalSlabs = Math.ceil(nz / slabSlices);

            // Center-out slab order
            const centerSlab = Math.min(Math.floor(nz / 2 / slabSlices), totalSlabs - 1);
            const slabOrder: number[] = [centerSlab];
            for (let d = 1; d < totalSlabs; d++) {
                if (centerSlab + d < totalSlabs) slabOrder.push(centerSlab + d);
                if (centerSlab - d >= 0) slabOrder.push(centerSlab - d);
            }

            for (const slabIdx of slabOrder) {
                // Cancellation check: user scrolled away
                if (this.currentXZIndex !== y) return;

                const zStart = slabIdx * slabSlices;
                const zEnd = Math.min(zStart + slabSlices, nz);

                // One large contiguous read for the slab
                const slabBuffer = await this.file.slice(
                    zStart * this.sliceBytes,
                    zEnd * this.sliceBytes,
                ).arrayBuffer();

                // Extract row y from each z-level using typed array views (zero-copy)
                for (let z = zStart; z < zEnd; z++) {
                    const i = z - zStart;
                    const rowByteOffset = (i * nx * ny + y * nx) * this.bpe;
                    const rowView = typedArrayView(slabBuffer, this.dataType, rowByteOffset, nx);
                    sliceData.set(rowView, z * nx);
                }

                // Update cached slice progressively
                if (this.currentXZIndex === y) {
                    this.currentXZSlice = {
                        data: sliceData,
                        width: nx,
                        height: nz,
                        isLowRes: true, // still partial
                    };
                    this.onSliceReady?.('xz', y);
                }

                await new Promise(r => setTimeout(r, 0));
            }

            // Final: mark as fully loaded
            if (this.currentXZIndex === y) {
                this.currentXZSlice = { data: sliceData, width: nx, height: nz };
                this.onSliceReady?.('xz', y);
            }
        } catch (e) {
            console.warn(`Failed to load XZ slice at y=${y}:`, e);
        } finally {
            this.xzLoadInProgress = false;
        }
    }

    // ========================================================================
    // YZ slice loading â€” independent async, cancellation, center-out slabs
    // ========================================================================

    /**
     * Async load high-res YZ slice using slab-based reads.
     * Column data is NOT contiguous, so reads slab of XY slices (~50MB),
     * creates typed array views per z-level, extracts column x.
     * Center-out order, cancellation, progressive callbacks.
     */
    private async loadYZSliceAsync(x: number): Promise<void> {
        if (this.yzLoadInProgress && this.currentYZIndex === x) return;

        const [nx, ny, nz] = this.dimensions;
        if (x < 0 || x >= nx) return;

        this.yzLoadInProgress = true;
        this.currentYZIndex = x;
        this.currentYZSlice = null;

        try {
            // Initialize with upscaled low-res data
            const sliceData = new Float32Array(ny * nz);
            const [lnx, lny, lnz] = this.lowRes.dimensions;
            const lowData = this.lowRes.data;
            const scale = Math.round(nx / lnx);
            const lx = Math.min(Math.floor(x / scale), lnx - 1);
            for (let z = 0; z < nz; z++) {
                const lz = Math.min(Math.floor(z / scale), lnz - 1);
                for (let yy = 0; yy < ny; yy++) {
                    const ly = Math.min(Math.floor(yy / scale), lny - 1);
                    sliceData[yy + z * ny] = lowData[lx + ly * lnx + lz * lnx * lny];
                }
            }

            // Slab sizing
            const slabSlices = Math.max(1, Math.floor(SLAB_BUDGET / this.sliceBytes));
            const totalSlabs = Math.ceil(nz / slabSlices);

            // Center-out slab order
            const centerSlab = Math.min(Math.floor(nz / 2 / slabSlices), totalSlabs - 1);
            const slabOrder: number[] = [centerSlab];
            for (let d = 1; d < totalSlabs; d++) {
                if (centerSlab + d < totalSlabs) slabOrder.push(centerSlab + d);
                if (centerSlab - d >= 0) slabOrder.push(centerSlab - d);
            }

            for (const slabIdx of slabOrder) {
                // Cancellation check
                if (this.currentYZIndex !== x) return;

                const zStart = slabIdx * slabSlices;
                const zEnd = Math.min(zStart + slabSlices, nz);

                const slabBuffer = await this.file.slice(
                    zStart * this.sliceBytes,
                    zEnd * this.sliceBytes,
                ).arrayBuffer();

                // Extract column x from each z-level
                for (let z = zStart; z < zEnd; z++) {
                    const i = z - zStart;
                    const sliceByteOffset = i * nx * ny * this.bpe;
                    const sliceView = typedArrayView(slabBuffer, this.dataType, sliceByteOffset, nx * ny);
                    for (let yy = 0; yy < ny; yy++) {
                        sliceData[yy + z * ny] = sliceView[x + yy * nx];
                    }
                }

                // Update cached slice progressively
                if (this.currentYZIndex === x) {
                    this.currentYZSlice = {
                        data: sliceData,
                        width: ny,
                        height: nz,
                        isLowRes: true,
                    };
                    this.onSliceReady?.('yz', x);
                }

                await new Promise(r => setTimeout(r, 0));
            }

            // Final: mark as fully loaded
            if (this.currentYZIndex === x) {
                this.currentYZSlice = { data: sliceData, width: ny, height: nz };
                this.onSliceReady?.('yz', x);
            }
        } catch (e) {
            console.warn(`Failed to load YZ slice at x=${x}:`, e);
        } finally {
            this.yzLoadInProgress = false;
        }
    }

    // ========================================================================
    // Low-res fallback
    // ========================================================================

    /** Get an upscaled low-res slice as fallback */
    private getLowResSlice(axis: ViewAxis, index: number): SliceData {
        const [nx, ny, nz] = this.dimensions;
        const [lnx, lny, lnz] = this.lowRes.dimensions;
        const lowData = this.lowRes.data;
        const scale = Math.round(nx / lnx);

        if (axis === 'xy') {
            const sliceData = new Float32Array(nx * ny);
            const lz = Math.min(Math.floor(index / scale), lnz - 1);
            for (let y = 0; y < ny; y++) {
                const ly = Math.min(Math.floor(y / scale), lny - 1);
                for (let x = 0; x < nx; x++) {
                    const lx = Math.min(Math.floor(x / scale), lnx - 1);
                    sliceData[x + y * nx] = lowData[lx + ly * lnx + lz * lnx * lny];
                }
            }
            return { data: sliceData, width: nx, height: ny, isLowRes: true };
        } else if (axis === 'xz') {
            const sliceData = new Float32Array(nx * nz);
            const ly = Math.min(Math.floor(index / scale), lny - 1);
            for (let z = 0; z < nz; z++) {
                const lz = Math.min(Math.floor(z / scale), lnz - 1);
                for (let x = 0; x < nx; x++) {
                    const lx = Math.min(Math.floor(x / scale), lnx - 1);
                    sliceData[x + z * nx] = lowData[lx + ly * lnx + lz * lnx * lny];
                }
            }
            return { data: sliceData, width: nx, height: nz, isLowRes: true };
        } else {
            const sliceData = new Float32Array(ny * nz);
            const lx = Math.min(Math.floor(index / scale), lnx - 1);
            for (let z = 0; z < nz; z++) {
                const lz = Math.min(Math.floor(z / scale), lnz - 1);
                for (let y = 0; y < ny; y++) {
                    const ly = Math.min(Math.floor(y / scale), lny - 1);
                    sliceData[y + z * ny] = lowData[lx + ly * lnx + lz * lnx * lny];
                }
            }
            return { data: sliceData, width: ny, height: nz, isLowRes: true };
        }
    }

    // ========================================================================
    // XY LRU cache
    // ========================================================================

    private putXYCache(index: number, slice: SliceData): void {
        if (this.xyCache.has(index)) {
            this.xyCache.delete(index);
        }
        this.xyCache.set(index, slice);

        while (this.xyCache.size > LRU_CACHE_SIZE) {
            const oldest = this.xyCache.keys().next().value;
            if (oldest !== undefined) {
                this.xyCache.delete(oldest);
            }
        }
    }

    /**
     * Check if 3D volume can be downsampled (dimensions > 1 on all axes).
     */
    canEnhance3D(): boolean {
        const [nx, ny, nz] = this.dimensions;
        return nx > 1 && ny > 1 && nz > 1;
    }

    /**
     * Create an anti-aliased downsampled 3D volume by streaming through the file.
     * @param scale Downsample factor (2 = half resolution, 4 = quarter)
     */
    async createDownsampledVolume(scale: number, onProgress?: (progress: number) => void): Promise<VolumeData | null> {
        const [nx, ny, nz] = this.dimensions;
        const dstNx = Math.ceil(nx / scale);
        const dstNy = Math.ceil(ny / scale);
        const dstNz = Math.ceil(nz / scale);

        console.log(`Downsampling 3D: Creating ${dstNx}x${dstNy}x${dstNz} volume (scale=${scale})`);

        const enhancedData = new Float32Array(dstNx * dstNy * dstNz);
        const xA = new Int32Array(dstNx);
        const xB = new Int32Array(dstNx);
        const xN = new Int32Array(dstNx);
        for (let dx = 0; dx < dstNx; dx++) {
            const x0 = dx * scale;
            const x1 = Math.min(x0 + scale, nx);
            const count = x1 - x0;
            xA[dx] = x0;
            xB[dx] = count > 1 ? (x1 - 1) : x0;
            xN[dx] = count > 1 ? 2 : 1;
        }
        const yA = new Int32Array(dstNy);
        const yB = new Int32Array(dstNy);
        const yN = new Int32Array(dstNy);
        for (let dy = 0; dy < dstNy; dy++) {
            const y0 = dy * scale;
            const y1 = Math.min(y0 + scale, ny);
            const count = y1 - y0;
            yA[dy] = y0;
            yB[dy] = count > 1 ? (y1 - 1) : y0;
            yN[dy] = count > 1 ? 2 : 1;
        }
        const xySampleCounts = new Int32Array(dstNx * dstNy);
        for (let dy = 0; dy < dstNy; dy++) {
            const row = dy * dstNx;
            for (let dx = 0; dx < dstNx; dx++) {
                xySampleCounts[row + dx] = xN[dx] * yN[dy];
            }
        }

        for (let dz = 0; dz < dstNz; dz++) {
            const z0 = dz * scale;
            const z1 = Math.min(z0 + scale, nz);
            const zCount = z1 - z0;
            const zN = zCount > 1 ? 2 : 1;
            const zFirst = z0;
            const zLast = zCount > 1 ? (z1 - 1) : z0;
            const accum = new Float64Array(dstNx * dstNy);

            for (let sz = 0; sz < zN; sz++) {
                const z = sz === 0 ? zFirst : zLast;
                const offset = z * this.sliceBytes;
                const blob = this.file.slice(offset, offset + this.sliceBytes);
                const buffer = await blob.arrayBuffer();
                const srcSlice = bufferToTypedArray(buffer, this.dataType, nx * ny);

                for (let dy = 0; dy < dstNy; dy++) {
                    const yFirst = yA[dy];
                    const yLast = yB[dy];
                    const yCount = yN[dy];
                    const outRow = dy * dstNx;

                    for (let dx = 0; dx < dstNx; dx++) {
                        const xFirst = xA[dx];
                        const xLast = xB[dx];
                        const xCount = xN[dx];
                        let sum2D = 0;
                        for (let sy = 0; sy < yCount; sy++) {
                            const y = sy === 0 ? yFirst : yLast;
                            const rowOffset = y * nx;
                            sum2D += srcSlice[rowOffset + xFirst];
                            if (xCount > 1) sum2D += srcSlice[rowOffset + xLast];
                        }
                        accum[outRow + dx] += sum2D;
                    }
                }
            }

            const dstZOffset = dz * dstNx * dstNy;
            for (let i = 0; i < accum.length; i++) {
                enhancedData[dstZOffset + i] = accum[i] / (xySampleCounts[i] * zN);
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
                spacing: [
                    this.spacing[0] * scale,
                    this.spacing[1] * scale,
                    this.spacing[2] * scale,
                ],
                dataType: 'float32'
            }
        );
    }
}

