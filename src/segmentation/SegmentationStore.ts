import { createMaskVolume } from './MaskVolume.js';
import type { MaskVolume } from './MaskTypes.js';
import { OpsQueue, type SegmentationOp } from './OpsQueue.js';
import {
    binaryMaskRLEToJSON,
    decodeLabelValuesRLE,
    encodeClassMaskRLE,
    encodeLabelValuesRLE,
    labelChunkRLEToJSON,
    type BinaryMaskRLEJson,
    type LabelChunkRLEJson,
} from './MaskPersistence.js';

interface EnsureMaskOptions {
    classDataType?: 'uint8' | 'uint16';
    preferredBackend?: 'auto' | 'dense' | 'sparse';
    maxDenseVoxels?: number;
    chunkSize?: number;
}

export interface SegmentationTileRLE {
    coord: [number, number, number];
    size: [number, number, number];
    data: LabelChunkRLEJson;
}

export interface SegmentationTilesSnapshot {
    format: 'viewer-segmentation-tiles-v1';
    dimensions: [number, number, number];
    classDataType: 'uint8' | 'uint16';
    chunkSize: number;
    tiles: SegmentationTileRLE[];
}

/**
 * Owns segmentation mask storage and operation history.
 * Keeps edit state lifetime and undo/redo independent from UI code.
 */
export class SegmentationStore {
    private _maskVolume: MaskVolume | null = null;
    private readonly opsQueue: OpsQueue;

    constructor(maxUndoDepth = 512) {
        this.opsQueue = new OpsQueue(maxUndoDepth);
    }

    get maskVolume(): MaskVolume | null {
        return this._maskVolume;
    }

    ensureMaskVolume(
        dimensions: [number, number, number],
        options: EnsureMaskOptions = {},
    ): boolean {
        const [nx, ny, nz] = dimensions;
        const sameDims = this._maskVolume
            && this._maskVolume.dimensions[0] === nx
            && this._maskVolume.dimensions[1] === ny
            && this._maskVolume.dimensions[2] === nz;
        if (sameDims) {
            return false;
        }

        const oldMask = this._maskVolume;
        this._maskVolume = null;
        this.opsQueue.clear();

        try {
            this._maskVolume = createMaskVolume(dimensions, {
                classDataType: options.classDataType ?? 'uint8',
                preferredBackend: options.preferredBackend ?? 'auto',
                maxDenseVoxels: options.maxDenseVoxels,
                chunkSize: options.chunkSize,
            });
        } catch (e) {
            this._maskVolume = null;
            oldMask?.dispose();
            throw e;
        }

        oldMask?.dispose();
        return true;
    }

    clearMask(): void {
        this._maskVolume?.dispose();
        this._maskVolume = null;
        this.opsQueue.clear();
    }

    clearOps(): void {
        this.opsQueue.clear();
    }

    applyOp(op: SegmentationOp): number {
        if (!this._maskVolume) return 0;
        return this.opsQueue.apply(op, { mask: this._maskVolume }).changedVoxels;
    }

    undo(): number {
        if (!this._maskVolume) return 0;
        return this.opsQueue.undo({ mask: this._maskVolume }).changedVoxels;
    }

    redo(): number {
        if (!this._maskVolume) return 0;
        return this.opsQueue.redo({ mask: this._maskVolume }).changedVoxels;
    }

    canUndo(): boolean {
        return this.opsQueue.canUndo();
    }

    canRedo(): boolean {
        return this.opsQueue.canRedo();
    }

    forEachVoxelOfClass(classId: number, visitor: (x: number, y: number, z: number, linearIndex: number) => void): number {
        if (!this._maskVolume) return 0;
        const [nx, ny] = this._maskVolume.dimensions;
        let matched = 0;
        this._maskVolume.forEachVoxelOfClass(classId, (x, y, z) => {
            const linear = x + y * nx + z * nx * ny;
            matched++;
            visitor(x, y, z, linear);
        });
        return matched;
    }

    remapClass(sourceClassId: number, targetClassId: number): number {
        if (!this._maskVolume) return 0;
        return this._maskVolume.remapClass(sourceClassId, targetClassId);
    }

    collectClassLinearIndices(classId: number): number[] {
        if (!this._maskVolume) return [];
        const indices: number[] = [];
        this.forEachVoxelOfClass(classId, (_x, _y, _z, linear) => {
            indices.push(linear);
        });
        return indices;
    }

    collectUsedClassIds(): number[] {
        if (!this._maskVolume) return [];
        const [nx, ny, nz] = this._maskVolume.dimensions;
        const used = new Set<number>();
        for (let z = 0; z < nz; z++) {
            for (let y = 0; y < ny; y++) {
                for (let x = 0; x < nx; x++) {
                    const value = this._maskVolume.getVoxel(x, y, z);
                    if (value > 0) used.add(value);
                }
            }
        }
        return Array.from(used).sort((a, b) => a - b);
    }

    serializeClassMaskRLE(classId: number): BinaryMaskRLEJson | null {
        if (!this._maskVolume) return null;
        return binaryMaskRLEToJSON(encodeClassMaskRLE(this._maskVolume, classId));
    }

    buildClassMaskBits(classId: number): Uint8Array | null {
        if (!this._maskVolume) return null;
        const [nx, ny, nz] = this._maskVolume.dimensions;
        const bits = new Uint8Array(nx * ny * nz);
        this.forEachVoxelOfClass(classId, (_x, _y, _z, linear) => {
            bits[linear] = 1;
        });
        return bits;
    }

    applyBinaryMaskBitsToClass(classId: number, bits: Uint8Array): number {
        if (!this._maskVolume) return 0;
        const [nx, ny, nz] = this._maskVolume.dimensions;
        const total = nx * ny * nz;
        if (bits.length !== total) {
            throw new Error(`Mask bits length mismatch: expected ${total}, got ${bits.length}`);
        }

        let changed = 0;
        let linear = 0;
        for (let z = 0; z < nz; z++) {
            for (let y = 0; y < ny; y++) {
                for (let x = 0; x < nx; x++, linear++) {
                    if (bits[linear] === 0) continue;
                    if (this._maskVolume.setVoxel(x, y, z, classId)) {
                        changed++;
                    }
                }
            }
        }
        return changed;
    }

    serializeLabelTiles(chunkSize = 64): SegmentationTilesSnapshot | null {
        if (!this._maskVolume) return null;

        const [nx, ny, nz] = this._maskVolume.dimensions;
        const cs = Math.max(8, Math.floor(chunkSize));
        const tiles: SegmentationTileRLE[] = [];
        const maxCx = Math.ceil(nx / cs);
        const maxCy = Math.ceil(ny / cs);
        const maxCz = Math.ceil(nz / cs);

        for (let cz = 0; cz < maxCz; cz++) {
            const z0 = cz * cs;
            const sz = Math.min(cs, nz - z0);
            for (let cy = 0; cy < maxCy; cy++) {
                const y0 = cy * cs;
                const sy = Math.min(cs, ny - y0);
                for (let cx = 0; cx < maxCx; cx++) {
                    const x0 = cx * cs;
                    const sx = Math.min(cs, nx - x0);
                    const count = sx * sy * sz;
                    const values = this._maskVolume.classDataType === 'uint16'
                        ? new Uint16Array(count)
                        : new Uint8Array(count);
                    let nonZero = 0;
                    let write = 0;
                    for (let z = z0; z < z0 + sz; z++) {
                        for (let y = y0; y < y0 + sy; y++) {
                            for (let x = x0; x < x0 + sx; x++, write++) {
                                const value = this._maskVolume.getVoxel(x, y, z);
                                values[write] = value;
                                if (value !== 0) nonZero++;
                            }
                        }
                    }
                    if (nonZero === 0) continue;
                    const encoded = encodeLabelValuesRLE(values);
                    tiles.push({
                        coord: [cx, cy, cz],
                        size: [sx, sy, sz],
                        data: labelChunkRLEToJSON(encoded),
                    });
                }
            }
        }

        return {
            format: 'viewer-segmentation-tiles-v1',
            dimensions: [nx, ny, nz],
            classDataType: this._maskVolume.classDataType,
            chunkSize: cs,
            tiles,
        };
    }

    restoreLabelTiles(snapshot: SegmentationTilesSnapshot, options: { clearFirst?: boolean } = {}): number {
        if (!this._maskVolume) return 0;

        const [nx, ny, nz] = this._maskVolume.dimensions;
        const [sx, sy, sz] = snapshot.dimensions;
        if (nx !== sx || ny !== sy || nz !== sz) {
            throw new Error(`Tile snapshot dimensions mismatch: expected ${nx}x${ny}x${nz}, got ${sx}x${sy}x${sz}`);
        }

        if (options.clearFirst !== false) {
            this._maskVolume.clear();
        }

        const cs = Math.max(8, Math.floor(snapshot.chunkSize || 64));
        let changed = 0;
        for (const tile of snapshot.tiles) {
            const [cx, cy, cz] = tile.coord;
            const [tileNx, tileNy, tileNz] = tile.size;
            const x0 = cx * cs;
            const y0 = cy * cs;
            const z0 = cz * cs;

            const expected = tileNx * tileNy * tileNz;
            const decoded = decodeLabelValuesRLE(
                {
                    totalVoxels: tile.data.totalVoxels,
                    values: tile.data.values,
                    lengths: tile.data.lengths,
                },
                this._maskVolume.classDataType,
            );
            if (decoded.length !== expected) {
                throw new Error(
                    `Decoded tile voxel count mismatch at ${cx},${cy},${cz}: expected ${expected}, got ${decoded.length}`,
                );
            }

            let read = 0;
            for (let lz = 0; lz < tileNz; lz++) {
                const gz = z0 + lz;
                if (gz < 0 || gz >= nz) {
                    read += tileNx * tileNy;
                    continue;
                }
                for (let ly = 0; ly < tileNy; ly++) {
                    const gy = y0 + ly;
                    if (gy < 0 || gy >= ny) {
                        read += tileNx;
                        continue;
                    }
                    for (let lx = 0; lx < tileNx; lx++, read++) {
                        const gx = x0 + lx;
                        if (gx < 0 || gx >= nx) continue;
                        const value = decoded[read];
                        if (value === 0) continue;
                        if (this._maskVolume.setVoxel(gx, gy, gz, value)) {
                            changed++;
                        }
                    }
                }
            }
        }
        this.opsQueue.clear();
        return changed;
    }
}
