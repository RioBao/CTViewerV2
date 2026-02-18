import type { MaskTypedArray, ViewAxis } from '../types.js';
import type { CreateMaskOptions, MaskBackend, MaskClassDataType, MaskSliceData, MaskVolume, SliceSelectionResult } from './MaskTypes.js';

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
    protected readonly totalVoxels: number;
    protected classCounts: Uint32Array;
    protected nonZeroVoxels = 0;
    protected _disposed = false;
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
        this.totalVoxels = dimensions[0] * dimensions[1] * dimensions[2];
        this.classCounts = new Uint32Array(this.maxClassId + 1);
        this.classCounts[0] = this.totalVoxels;
        this.backend = backend;
    }

    protected checkDisposed(): void {
        if (this._disposed) throw new Error('MaskVolume has been disposed');
    }

    abstract getVoxel(x: number, y: number, z: number): number;
    abstract setVoxel(x: number, y: number, z: number, classId: number): boolean;
    abstract forEachVoxelOfClass(classId: number, visitor: (x: number, y: number, z: number) => void): number;
    abstract remapClass(sourceClassId: number, targetClassId: number): number;
    abstract writeSliceSelection(axis: ViewAxis, sliceIndex: number, width: number, selectedIndices: Uint32Array, classId: number): SliceSelectionResult;
    abstract restoreLinearValues(linear: Uint32Array, before: MaskTypedArray): number;
    abstract clear(): void;
    abstract fill(classId: number): void;
    abstract getSlice(axis: ViewAxis, index: number, target?: MaskTypedArray): MaskSliceData;
    abstract dispose(): void;

    getNonZeroVoxelCount(): number {
        this.checkDisposed();
        return this.nonZeroVoxels;
    }

    getClassVoxelCount(classId: number): number {
        this.checkDisposed();
        const normalized = this.normalizeClassIdOrNull(classId);
        if (normalized == null) return 0;
        return this.classCounts[normalized];
    }

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

    protected applyClassRemap(src: number, tgt: number, count: number): void {
        this.classCounts[src] -= count;
        this.classCounts[tgt] += count;
        if (src === 0) this.nonZeroVoxels += count;
        else if (tgt === 0) this.nonZeroVoxels -= count;
    }

    protected normalizeClassIdOrNull(classId: number): number | null {
        if (!Number.isFinite(classId)) return null;
        const rounded = Math.round(classId);
        if (rounded < 0 || rounded > this.maxClassId) return null;
        return rounded;
    }

    protected resetClassCounts(fillClass: number): void {
        this.classCounts.fill(0);
        this.classCounts[fillClass] = this.totalVoxels;
        this.nonZeroVoxels = fillClass === 0 ? 0 : this.totalVoxels;
    }

    protected noteVoxelClassChange(previousClass: number, nextClass: number): void {
        if (previousClass === nextClass) return;
        this.classCounts[previousClass]--;
        this.classCounts[nextClass]++;
        if (previousClass === 0 && nextClass !== 0) {
            this.nonZeroVoxels++;
        } else if (previousClass !== 0 && nextClass === 0) {
            this.nonZeroVoxels--;
        }
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
        this.checkDisposed();
        const [nx, ny, nz] = this.dimensions;
        if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) return 0;
        return this.data[x + y * nx + z * nx * ny];
    }

    setVoxel(x: number, y: number, z: number, classId: number): boolean {
        this.checkDisposed();
        const [nx, ny, nz] = this.dimensions;
        if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) return false;
        const idx = x + y * nx + z * nx * ny;
        const next = clampClassId(classId, this.maxClassId);
        const previous = this.data[idx];
        if (previous === next) return false;
        this.data[idx] = next;
        this.noteVoxelClassChange(previous, next);
        this.markVoxelDirty(x, y, z);
        return true;
    }

    forEachVoxelOfClass(classId: number, visitor: (x: number, y: number, z: number) => void): number {
        this.checkDisposed();
        const target = this.normalizeClassIdOrNull(classId);
        if (target == null) return 0;
        const expected = this.classCounts[target];
        if (expected === 0) return 0;

        const [nx, ny, nz] = this.dimensions;
        let matched = 0;
        let linear = 0;
        for (let z = 0; z < nz; z++) {
            for (let y = 0; y < ny; y++) {
                for (let x = 0; x < nx; x++, linear++) {
                    if (this.data[linear] !== target) continue;
                    visitor(x, y, z);
                    matched++;
                }
            }
        }
        return matched;
    }

    remapClass(sourceClassId: number, targetClassId: number): number {
        this.checkDisposed();
        const src = this.normalizeClassIdOrNull(sourceClassId);
        const tgt = this.normalizeClassIdOrNull(targetClassId);
        if (src == null || tgt == null || src === tgt) return 0;
        if (this.classCounts[src] === 0) return 0;

        // Scan the flat typed array directly — no per-voxel bounds checks or
        // coordinate arithmetic, so JS engines can optimise this inner loop well.
        let changed = 0;
        for (let i = 0; i < this.data.length; i++) {
            if (this.data[i] === src) {
                this.data[i] = tgt;
                changed++;
            }
        }
        if (changed > 0) {
            this.applyClassRemap(src, tgt, changed);
            this.finalizeBulkEdit();
        }
        return changed;
    }

    writeSliceSelection(axis: ViewAxis, sliceIndex: number, width: number, selectedIndices: Uint32Array, classId: number): SliceSelectionResult {
        this.checkDisposed();
        const tgt = clampClassId(classId, this.maxClassId);
        const [nx, ny] = this.dimensions;
        const nxny = nx * ny;

        // Pre-allocate worst-case undo arrays (compact, typed — no per-voxel objects).
        const undoLinear = new Uint32Array(selectedIndices.length);
        const undoBefore = createMaskArray(selectedIndices.length, this.classDataType);
        let changed = 0;

        for (let i = 0; i < selectedIndices.length; i++) {
            const s = selectedIndices[i];
            const sliceX = s % width;
            const sliceY = (s / width) | 0;

            // Map 2D slice coords to flat 3D linear index — no bounds-check per pixel,
            // no virtual dispatch, no clampClassId on every iteration.
            let linear: number;
            switch (axis) {
                case 'xy': linear = sliceX       + sliceY * nx + sliceIndex * nxny; break;
                case 'xz': linear = sliceX       + sliceIndex * nx + sliceY * nxny; break;
                default:   linear = sliceIndex   + sliceX * nx    + sliceY * nxny; break; // yz
            }
            if (linear < 0 || linear >= this.data.length) continue;

            const before = this.data[linear];
            if (before === tgt) continue;

            this.data[linear] = tgt;
            this.noteVoxelClassChange(before, tgt);
            undoLinear[changed] = linear;
            undoBefore[changed] = before;
            changed++;
        }

        if (changed > 0) this.finalizeBulkEdit(); // markAllDirty once, not per voxel
        return { changed, undoLinear: undoLinear.subarray(0, changed), undoBefore: undoBefore.subarray(0, changed) };
    }

    restoreLinearValues(linear: Uint32Array, before: MaskTypedArray): number {
        this.checkDisposed();
        let changed = 0;
        for (let i = 0; i < linear.length; i++) {
            const idx = linear[i];
            const prev = this.data[idx];
            const next = before[i];
            if (prev === next) continue;
            this.data[idx] = next;
            this.noteVoxelClassChange(prev, next);
            changed++;
        }
        if (changed > 0) this.finalizeBulkEdit();
        return changed;
    }

    clear(): void {
        this.checkDisposed();
        this.data.fill(0);
        this.resetClassCounts(0);
        this.finalizeBulkEdit();
    }

    fill(classId: number): void {
        this.checkDisposed();
        const fillClass = clampClassId(classId, this.maxClassId);
        this.data.fill(fillClass);
        this.resetClassCounts(fillClass);
        this.finalizeBulkEdit();
    }

    getSlice(axis: ViewAxis, index: number, target?: MaskTypedArray): MaskSliceData {
        this.checkDisposed();
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
        this._disposed = true;
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
    /** Lazy fill value - when non-zero, voxels without a chunk return this instead of 0. */
    private fillValue = 0;

    constructor(dimensions: [number, number, number], classDataType: MaskClassDataType, chunkSize: number) {
        super(dimensions, classDataType, 'sparse');
        this.chunkSize = Math.max(8, Math.min(128, Math.floor(chunkSize)));
    }

    getVoxel(x: number, y: number, z: number): number {
        this.checkDisposed();
        const [nx, ny, nz] = this.dimensions;
        if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) return 0;

        const cs = this.chunkSize;
        const cx = Math.floor(x / cs);
        const cy = Math.floor(y / cs);
        const cz = Math.floor(z / cs);
        const key = packChunkKey(cx, cy, cz);
        const chunk = this.chunks.get(key);
        if (!chunk) return this.fillValue;

        const lx = x - cx * cs;
        const ly = y - cy * cs;
        const lz = z - cz * cs;
        return chunk[lx + ly * cs + lz * cs * cs];
    }

    setVoxel(x: number, y: number, z: number, classId: number): boolean {
        this.checkDisposed();
        const [nx, ny, nz] = this.dimensions;
        if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) return false;

        const next = clampClassId(classId, this.maxClassId);
        const cs = this.chunkSize;
        const cx = Math.floor(x / cs);
        const cy = Math.floor(y / cs);
        const cz = Math.floor(z / cs);
        const key = packChunkKey(cx, cy, cz);
        const lx = x - cx * cs;
        const ly = y - cy * cs;
        const lz = z - cz * cs;
        const idx = lx + ly * cs + lz * cs * cs;

        let chunk = this.chunks.get(key);
        const previous = chunk ? chunk[idx] : this.fillValue;
        if (previous === next) return false;
        if (!chunk) {
            chunk = createMaskArray(cs * cs * cs, this.classDataType);
            if (this.fillValue !== 0) chunk.fill(this.fillValue);
            this.chunks.set(key, chunk);
        }
        chunk[idx] = next;
        this.noteVoxelClassChange(previous, next);

        this.markVoxelDirty(x, y, z);
        return true;
    }

    forEachVoxelOfClass(classId: number, visitor: (x: number, y: number, z: number) => void): number {
        this.checkDisposed();
        const target = this.normalizeClassIdOrNull(classId);
        if (target == null) return 0;
        const expected = this.classCounts[target];
        if (expected === 0) return 0;

        const [nx, ny, nz] = this.dimensions;
        const cs = this.chunkSize;
        let matched = 0;

        if (this.fillValue === target) {
            for (let z = 0; z < nz; z++) {
                for (let y = 0; y < ny; y++) {
                    for (let x = 0; x < nx; x++) {
                        if (this.getVoxel(x, y, z) !== target) continue;
                        visitor(x, y, z);
                        matched++;
                    }
                }
            }
            return matched;
        }

        for (const [key, chunk] of this.chunks) {
            const { cx, cy, cz } = unpackChunkKey(key);
            const x0 = cx * cs;
            const y0 = cy * cs;
            const z0 = cz * cs;
            const xMax = Math.min(x0 + cs, nx);
            const yMax = Math.min(y0 + cs, ny);
            const zMax = Math.min(z0 + cs, nz);

            for (let z = z0; z < zMax; z++) {
                const lz = z - z0;
                for (let y = y0; y < yMax; y++) {
                    const ly = y - y0;
                    const rowBase = lz * cs * cs + ly * cs;
                    for (let x = x0; x < xMax; x++) {
                        if (chunk[rowBase + (x - x0)] !== target) continue;
                        visitor(x, y, z);
                        matched++;
                    }
                }
            }
        }

        return matched;
    }

    remapClass(sourceClassId: number, targetClassId: number): number {
        this.checkDisposed();
        const src = this.normalizeClassIdOrNull(sourceClassId);
        const tgt = this.normalizeClassIdOrNull(targetClassId);
        if (src == null || tgt == null || src === tgt) return 0;
        if (this.classCounts[src] === 0) return 0;

        // For sparse volumes the key advantage: only allocated chunks are scanned.
        // Unallocated voxels implicitly hold fillValue (always 0 in normal use);
        // if fillValue happens to equal src, remap it too without touching chunks.
        let changed = 0;

        if (this.fillValue === src) {
            // All implicit (unallocated) voxels currently hold src — remap them by
            // updating fillValue. The count is tracked in classCounts already.
            this.fillValue = tgt;
            changed += this.classCounts[src]; // includes implicit voxels
        }

        for (const chunk of this.chunks.values()) {
            for (let i = 0; i < chunk.length; i++) {
                if (chunk[i] === src) {
                    chunk[i] = tgt;
                    changed++;
                }
            }
        }

        if (changed > 0) {
            this.applyClassRemap(src, tgt, changed);
            this.finalizeBulkEdit();
        }
        return changed;
    }

    writeSliceSelection(axis: ViewAxis, sliceIndex: number, width: number, selectedIndices: Uint32Array, classId: number): SliceSelectionResult {
        this.checkDisposed();
        const tgt = clampClassId(classId, this.maxClassId);
        const [nx, ny, nz] = this.dimensions;
        const cs = this.chunkSize;
        const nxny = nx * ny;

        const undoLinear = new Uint32Array(selectedIndices.length);
        const undoBefore = createMaskArray(selectedIndices.length, this.classDataType);
        let changed = 0;

        for (let i = 0; i < selectedIndices.length; i++) {
            const s = selectedIndices[i];
            const sliceX = s % width;
            const sliceY = (s / width) | 0;

            let x: number, y: number, z: number, linear: number;
            switch (axis) {
                case 'xy': x = sliceX; y = sliceY; z = sliceIndex; linear = x + y * nx + z * nxny; break;
                case 'xz': x = sliceX; y = sliceIndex; z = sliceY; linear = x + y * nx + z * nxny; break;
                default:   x = sliceIndex; y = sliceX; z = sliceY; linear = x + y * nx + z * nxny; break; // yz
            }
            if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) continue;

            // Directly access the chunk, allocating it if needed.
            const cx = Math.floor(x / cs);
            const cy = Math.floor(y / cs);
            const cz = Math.floor(z / cs);
            const key = packChunkKey(cx, cy, cz);
            let chunk = this.chunks.get(key);
            const lx = x - cx * cs;
            const ly = y - cy * cs;
            const lz = z - cz * cs;
            const idx = lx + ly * cs + lz * cs * cs;
            const before = chunk ? chunk[idx] : this.fillValue;

            if (before === tgt) continue;

            if (!chunk) {
                chunk = createMaskArray(cs * cs * cs, this.classDataType);
                if (this.fillValue !== 0) chunk.fill(this.fillValue);
                this.chunks.set(key, chunk);
            }
            chunk[idx] = tgt;
            this.noteVoxelClassChange(before, tgt);
            undoLinear[changed] = linear;
            undoBefore[changed] = before;
            changed++;
        }

        if (changed > 0) this.finalizeBulkEdit();
        return { changed, undoLinear: undoLinear.subarray(0, changed), undoBefore: undoBefore.subarray(0, changed) };
    }

    restoreLinearValues(linear: Uint32Array, before: MaskTypedArray): number {
        this.checkDisposed();
        const [nx, , ] = this.dimensions;
        const nxny = nx * this.dimensions[1];
        const cs = this.chunkSize;
        let changed = 0;

        for (let i = 0; i < linear.length; i++) {
            const lin = linear[i];
            const x = lin % nx;
            const y = ((lin / nx) | 0) % this.dimensions[1];
            const z = (lin / nxny) | 0;

            const cx = Math.floor(x / cs);
            const cy = Math.floor(y / cs);
            const cz = Math.floor(z / cs);
            const key = packChunkKey(cx, cy, cz);
            let chunk = this.chunks.get(key);
            const lx = x - cx * cs;
            const ly = y - cy * cs;
            const lz = z - cz * cs;
            const idx = lx + ly * cs + lz * cs * cs;
            const prev = chunk ? chunk[idx] : this.fillValue;
            const next = before[i];

            if (prev === next) continue;
            if (!chunk) {
                chunk = createMaskArray(cs * cs * cs, this.classDataType);
                if (this.fillValue !== 0) chunk.fill(this.fillValue);
                this.chunks.set(key, chunk);
            }
            chunk[idx] = next;
            this.noteVoxelClassChange(prev, next);
            changed++;
        }

        if (changed > 0) this.finalizeBulkEdit();
        return changed;
    }

    clear(): void {
        this.checkDisposed();
        this.chunks.clear();
        this.fillValue = 0;
        this.resetClassCounts(0);
        this.finalizeBulkEdit();
    }

    fill(classId: number): void {
        this.checkDisposed();
        const fillClass = clampClassId(classId, this.maxClassId);
        this.chunks.clear();
        this.fillValue = fillClass;
        this.resetClassCounts(fillClass);
        this.finalizeBulkEdit();
    }

    getSlice(axis: ViewAxis, index: number, target?: MaskTypedArray): MaskSliceData {
        this.checkDisposed();
        const [w, h] = sliceShape(axis, this.dimensions);
        const len = w * h;
        let out = target;
        if (!out || out.length !== len) {
            out = createMaskArray(len, this.classDataType);
        }

        // Fill with the lazy fill value (0 leaves the buffer zeroed after allocation)
        if (this.fillValue !== 0) {
            out.fill(this.fillValue);
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
        this._disposed = true;
        this.chunks.clear();
        this.fillValue = 0;
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
