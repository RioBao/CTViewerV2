import type { MaskTypedArray, ViewAxis } from '../types.js';

export type MaskClassDataType = 'uint8' | 'uint16';
export type MaskBackend = 'dense' | 'sparse';

export interface MaskSliceData {
    data: MaskTypedArray;
    width: number;
    height: number;
}

/** Compact result of a bulk slice-selection write, used for undo. */
export interface SliceSelectionResult {
    changed: number;
    /** Linear voxel indices (in 3D flat layout) of every voxel that changed. */
    undoLinear: Uint32Array;
    /** Previous class id of each changed voxel; same length as undoLinear. */
    undoBefore: MaskTypedArray;
}

export interface MaskVolume {
    readonly dimensions: [number, number, number];
    readonly classDataType: MaskClassDataType;
    readonly maxClassId: number;
    readonly backend: MaskBackend;
    getVoxel(x: number, y: number, z: number): number;
    setVoxel(x: number, y: number, z: number, classId: number): boolean;
    getNonZeroVoxelCount(): number;
    getClassVoxelCount(classId: number): number;
    forEachVoxelOfClass(classId: number, visitor: (x: number, y: number, z: number) => void): number;
    remapClass(sourceClassId: number, targetClassId: number): number;
    /**
     * Write `classId` to every voxel whose 2D index within the slice appears in
     * `selectedIndices`. Returns compact flat-array undo data (no per-voxel
     * JS objects) and calls markAllDirty once rather than per voxel.
     */
    writeSliceSelection(axis: ViewAxis, sliceIndex: number, width: number, selectedIndices: Uint32Array, classId: number): SliceSelectionResult;
    /** Restore voxels to previous class ids — the undo counterpart of writeSliceSelection. */
    restoreLinearValues(linear: Uint32Array, before: MaskTypedArray): number;
    clear(): void;
    fill(classId: number): void;
    getSlice(axis: ViewAxis, index: number, target?: MaskTypedArray): MaskSliceData;
    consumeSliceDirty(axis: ViewAxis, index: number): boolean;
    markAllDirty(): void;
    dispose(): void;
}

export interface CreateMaskOptions {
    preferredBackend?: 'auto' | MaskBackend;
    classDataType?: MaskClassDataType;
    maxDenseVoxels?: number;
    chunkSize?: number;
}
