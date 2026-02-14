import type { MaskTypedArray, ViewAxis } from '../types.js';
import type { CreateMaskOptions, MaskBackend, MaskClassDataType, MaskSliceData, MaskVolume } from './MaskTypes.js';

const DEFAULT_MAX_DENSE_VOXELS = 128 * 1024 * 1024;
const DEFAULT_CHUNK_SIZE = 64;

function clampClassId(classId: number, maxClassId: number): number {
    if (!Number.isFinite(classId)) return 0;
    const rounded = Math.round(classId);
    if (rounded < 0) return 0;
    if (rounded > maxClassId) return maxClassId;
    return rounded;
}

function createMaskArray(length: number, dt: MaskClassDataType): MaskTypedArray {
    return dt === 'uint16' ? new Uint16Array(length) : new Uint8Array(length);
}

function sliceShape(axis: ViewAxis, dims: [number, number, number]): [number, number] {
    const [nx, ny, nz] = dims;
    switch (axis) {
        case 'xy': return [nx, ny];
        case 'xz': return [nx, nz];
        case 'yz': return [ny, nz];
    }
}

abstract class BaseMaskVolume implements MaskVolume {
    readonly dimensions: [number, number, number];
    readonly classDataType: MaskClassDataType;
    readonly maxClassId: number;
    readonly backend: MaskBackend;
    protected dirtyAll = true;
    protected dirtyXY = new Set<number>();
    protected dirtyXZ = new Set<number>();
    protected dirtyYZ = new Set<number>();

    protected constructor(
        dimensions: [number, number, number],
        classDataType: MaskClassDataType,
        backend: MaskBackend,
    ) {
        this.dimensions = dimensions;
        this.classDataType = classDataType;
        this.maxClassId = classDataType === 'uint16' ? 65535 : 255;
        this.backend = backend;
    }

    abstract getVoxel(x: number, y: number, z: number): number;
    abstract setVoxel(x: number, y: number, z: number, classId: number): boolean;
    abstract clear(): void;
    abstract fill(classId: number): void;
    abstract getSlice(axis: ViewAxis, index: number, target?: MaskTypedArray): MaskSliceData;
    abstract dispose(): void;

    consumeSliceDirty(axis: ViewAxis, index: number): boolean {
        if (this.dirtyAll) return true;
        switch (axis) {
            case 'xy': return this.dirtyXY.delete(index);
            case 'xz': return this.dirtyXZ.delete(index);
            case 'yz': return this.dirtyYZ.delete(index);
        }
    }

    markAllDirty(): void {
        this.dirtyAll = true;
        this.dirtyXY.clear();
        this.dirtyXZ.clear();
        this.dirtyYZ.clear();
    }

    protected markVoxelDirty(x: number, y: number, z: number): void {
        if (this.dirtyAll) return;
        this.dirtyXY.add(z);
        this.dirtyXZ.add(y);
        this.dirtyYZ.add(x);
    }

    protected finalizeBulkEdit(): void {
        this.markAllDirty();
    }
}

class DenseMaskVolume extends BaseMaskVolume {
    private data: MaskTypedArray;

    constructor(dimensions: [number, number, number], classDataType: MaskClassDataType) {
        super(dimensions, classDataType, 'dense');
        const [nx, ny, nz] = dimensions;
        this.data = createMaskArray(nx * ny * nz, classDataType);
    }

    getVoxel(x: number, y: number, z: number): number {
        const [nx, ny, nz] = this.dimensions;
        if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) return 0;
        return this.data[x + y * nx + z * nx * ny];
    }

    setVoxel(x: number, y: number, z: number, classId: number): boolean {
        const [nx, ny, nz] = this.dimensions;
        if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) return false;
        const idx = x + y * nx + z * nx * ny;
        const next = clampClassId(classId, this.maxClassId);
        if (this.data[idx] === next) return false;
        this.data[idx] = next;
        this.markVoxelDirty(x, y, z);
        return true;
    }

    clear(): void {
        this.data.fill(0);
        this.finalizeBulkEdit();
    }

    fill(classId: number): void {
        this.data.fill(clampClassId(classId, this.maxClassId));
        this.finalizeBulkEdit();
    }

    getSlice(axis: ViewAxis, index: number, target?: MaskTypedArray): MaskSliceData {
        const [w, h] = sliceShape(axis, this.dimensions);
        const len = w * h;
        let out = target;
        if (!out || out.length !== len || out.constructor !== this.data.constructor) {
            out = createMaskArray(len, this.classDataType);
        }

        const [nx, ny, nz] = this.dimensions;
        switch (axis) {
            case 'xy': {
                if (index < 0 || index >= nz) throw new Error(`Mask slice index ${index} out of bounds for axis xy`);
                const offset = index * nx * ny;
                out.set(this.data.subarray(offset, offset + nx * ny));
                break;
            }
            case 'xz': {
                if (index < 0 || index >= ny) throw new Error(`Mask slice index ${index} out of bounds for axis xz`);
                for (let z = 0; z < nz; z++) {
                    const srcBase = index * nx + z * nx * ny;
                    const dstBase = z * nx;
                    out.set(this.data.subarray(srcBase, srcBase + nx), dstBase);
                }
                break;
            }
            case 'yz': {
                if (index < 0 || index >= nx) throw new Error(`Mask slice index ${index} out of bounds for axis yz`);
                for (let z = 0; z < nz; z++) {
                    const srcBase = z * nx * ny + index;
                    const dstBase = z * ny;
                    for (let y = 0; y < ny; y++) {
                        out[dstBase + y] = this.data[srcBase + y * nx];
                    }
                }
                break;
            }
        }

        this.dirtyAll = false;
        return { data: out, width: w, height: h };
    }

    dispose(): void {
        this.data = createMaskArray(0, this.classDataType);
    }
}

interface ChunkKey {
    cx: number;
    cy: number;
    cz: number;
}

function packChunkKey(cx: number, cy: number, cz: number): string {
    return `${cx},${cy},${cz}`;
}

function unpackChunkKey(key: string): ChunkKey {
    const parts = key.split(',');
    return {
        cx: Number(parts[0]),
        cy: Number(parts[1]),
        cz: Number(parts[2]),
    };
}

class SparseChunkMaskVolume extends BaseMaskVolume {
    private readonly chunkSize: number;
    private readonly chunks = new Map<string, MaskTypedArray>();

    constructor(dimensions: [number, number, number], classDataType: MaskClassDataType, chunkSize: number) {
        super(dimensions, classDataType, 'sparse');
        this.chunkSize = Math.max(8, Math.min(128, Math.floor(chunkSize)));
    }

    getVoxel(x: number, y: number, z: number): number {
        const [nx, ny, nz] = this.dimensions;
        if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) return 0;

        const cs = this.chunkSize;
        const cx = Math.floor(x / cs);
        const cy = Math.floor(y / cs);
        const cz = Math.floor(z / cs);
        const key = packChunkKey(cx, cy, cz);
        const chunk = this.chunks.get(key);
        if (!chunk) return 0;

        const lx = x - cx * cs;
        const ly = y - cy * cs;
        const lz = z - cz * cs;
        return chunk[lx + ly * cs + lz * cs * cs];
    }

    setVoxel(x: number, y: number, z: number, classId: number): boolean {
        const [nx, ny, nz] = this.dimensions;
        if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) return false;

        const next = clampClassId(classId, this.maxClassId);
        const cs = this.chunkSize;
        const cx = Math.floor(x / cs);
        const cy = Math.floor(y / cs);
        const cz = Math.floor(z / cs);
        const key = packChunkKey(cx, cy, cz);

        let chunk = this.chunks.get(key);
        if (!chunk) {
            if (next === 0) return false;
            chunk = createMaskArray(cs * cs * cs, this.classDataType);
            this.chunks.set(key, chunk);
        }

        const lx = x - cx * cs;
        const ly = y - cy * cs;
        const lz = z - cz * cs;
        const idx = lx + ly * cs + lz * cs * cs;
        if (chunk[idx] === next) return false;
        chunk[idx] = next;

        this.markVoxelDirty(x, y, z);
        return true;
    }

    clear(): void {
        this.chunks.clear();
        this.finalizeBulkEdit();
    }

    fill(classId: number): void {
        const fillClass = clampClassId(classId, this.maxClassId);
        this.chunks.clear();
        if (fillClass !== 0) {
            const [nx, ny, nz] = this.dimensions;
            const cs = this.chunkSize;
            const maxCx = Math.ceil(nx / cs);
            const maxCy = Math.ceil(ny / cs);
            const maxCz = Math.ceil(nz / cs);
            for (let cz = 0; cz < maxCz; cz++) {
                for (let cy = 0; cy < maxCy; cy++) {
                    for (let cx = 0; cx < maxCx; cx++) {
                        const chunk = createMaskArray(cs * cs * cs, this.classDataType);
                        chunk.fill(fillClass);
                        this.chunks.set(packChunkKey(cx, cy, cz), chunk);
                    }
                }
            }
        }
        this.finalizeBulkEdit();
    }

    getSlice(axis: ViewAxis, index: number, target?: MaskTypedArray): MaskSliceData {
        const [w, h] = sliceShape(axis, this.dimensions);
        const len = w * h;
        let out = target;
        if (!out || out.length !== len) {
            out = createMaskArray(len, this.classDataType);
        } else {
            out.fill(0);
        }

        const [nx, ny, nz] = this.dimensions;
        const cs = this.chunkSize;

        switch (axis) {
            case 'xy': {
                if (index < 0 || index >= nz) throw new Error(`Mask slice index ${index} out of bounds for axis xy`);
                const czTarget = Math.floor(index / cs);
                const lz = index - czTarget * cs;
                for (const [key, chunk] of this.chunks) {
                    const { cx, cy, cz } = unpackChunkKey(key);
                    if (cz !== czTarget) continue;
                    const x0 = cx * cs;
                    const y0 = cy * cs;
                    const xMax = Math.min(x0 + cs, nx);
                    const yMax = Math.min(y0 + cs, ny);
                    for (let y = y0; y < yMax; y++) {
                        const ly = y - y0;
                        const rowBaseOut = y * nx;
                        const rowBaseChunk = lz * cs * cs + ly * cs;
                        for (let x = x0; x < xMax; x++) {
                            out[rowBaseOut + x] = chunk[rowBaseChunk + (x - x0)];
                        }
                    }
                }
                break;
            }
            case 'xz': {
                if (index < 0 || index >= ny) throw new Error(`Mask slice index ${index} out of bounds for axis xz`);
                const cyTarget = Math.floor(index / cs);
                const ly = index - cyTarget * cs;
                for (const [key, chunk] of this.chunks) {
                    const { cx, cy, cz } = unpackChunkKey(key);
                    if (cy !== cyTarget) continue;
                    const x0 = cx * cs;
                    const z0 = cz * cs;
                    const xMax = Math.min(x0 + cs, nx);
                    const zMax = Math.min(z0 + cs, nz);
                    for (let z = z0; z < zMax; z++) {
                        const lz = z - z0;
                        const rowBaseOut = z * nx;
                        const rowBaseChunk = lz * cs * cs + ly * cs;
                        for (let x = x0; x < xMax; x++) {
                            out[rowBaseOut + x] = chunk[rowBaseChunk + (x - x0)];
                        }
                    }
                }
                break;
            }
            case 'yz': {
                if (index < 0 || index >= nx) throw new Error(`Mask slice index ${index} out of bounds for axis yz`);
                const cxTarget = Math.floor(index / cs);
                const lx = index - cxTarget * cs;
                for (const [key, chunk] of this.chunks) {
                    const { cx, cy, cz } = unpackChunkKey(key);
                    if (cx !== cxTarget) continue;
                    const y0 = cy * cs;
                    const z0 = cz * cs;
                    const yMax = Math.min(y0 + cs, ny);
                    const zMax = Math.min(z0 + cs, nz);
                    for (let z = z0; z < zMax; z++) {
                        const lz = z - z0;
                        const rowBaseOut = z * ny;
                        const rowBaseChunk = lz * cs * cs + lx;
                        for (let y = y0; y < yMax; y++) {
                            out[rowBaseOut + y] = chunk[rowBaseChunk + (y - y0) * cs];
                        }
                    }
                }
                break;
            }
        }

        this.dirtyAll = false;
        return { data: out, width: w, height: h };
    }

    dispose(): void {
        this.chunks.clear();
    }
}

export function createMaskVolume(
    dimensions: [number, number, number],
    options: CreateMaskOptions = {},
): MaskVolume {
    const classDataType = options.classDataType ?? 'uint8';
    const preferred = options.preferredBackend ?? 'auto';
    const maxDenseVoxels = options.maxDenseVoxels ?? DEFAULT_MAX_DENSE_VOXELS;
    const [nx, ny, nz] = dimensions;
    const totalVoxels = nx * ny * nz;

    let backend: MaskBackend;
    if (preferred === 'auto') {
        backend = totalVoxels <= maxDenseVoxels ? 'dense' : 'sparse';
    } else {
        backend = preferred;
    }

    if (backend === 'sparse') {
        return new SparseChunkMaskVolume(dimensions, classDataType, options.chunkSize ?? DEFAULT_CHUNK_SIZE);
    }
    return new DenseMaskVolume(dimensions, classDataType);
}
