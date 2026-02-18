import type { MaskTypedArray, ViewAxis } from '../types.js';

export type MaskClassDataType = 'uint8' | 'uint16';
export type MaskBackend = 'dense' | 'sparse';

export interface MaskSliceData {
    data: MaskTypedArray;
    width: number;
    height: number;
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
