/** Orthogonal slice axis */
export type ViewAxis = 'xy' | 'xz' | 'yz';

/** All viewport identifiers including 3D */
export type ViewId = ViewAxis | '3d';

/** Supported voxel data types */
export type VoxelDataType = 'uint8' | 'uint16' | 'float32';

/** TypedArray types used for voxel data */
export type VoxelTypedArray = Uint8Array | Uint16Array | Float32Array;

/** Volume metadata describing the loaded dataset */
export interface VolumeMetadata {
    dimensions: [number, number, number];
    dataType: VoxelDataType;
    spacing: [number, number, number];
    byteOrder?: 'little-endian' | 'big-endian';
    isRGB?: boolean;
    min?: number;
    max?: number;
    description?: string;
}

/** 2D slice extracted from a volume */
export interface SliceData {
    data: VoxelTypedArray;
    width: number;
    height: number;
    isLowRes?: boolean;
}
