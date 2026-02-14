/** Orthogonal slice axis */
export type ViewAxis = 'xy' | 'xz' | 'yz';

/** All viewport identifiers including 3D */
export type ViewId = ViewAxis | '3d';

/** Top-level interaction mode */
export enum AppMode {
    Viewing = 'viewing',
    Measuring = 'measuring',
    Segmentation = 'segmentation',
}

/** Supported voxel data types */
export type VoxelDataType = 'uint8' | 'uint16' | 'float32';

/** TypedArray types used for voxel data */
export type VoxelTypedArray = Uint8Array | Uint16Array | Float32Array;

/** TypedArray types used for segmentation mask data */
export type MaskTypedArray = Uint8Array | Uint16Array;

/** Segmentation tool identifier */
export type SegmentationTool = 'brush' | 'threshold' | 'region-grow';

/** Segmentation mode tool identifier */
export type SegmentationModeTool = 'brush' | 'erase' | 'threshold' | 'smart-region' | null;

/** Segmentation UI/runtime settings */
export interface SegmentationSettings {
    enabled: boolean;
    visible: boolean;
    overlayOpacity: number;
    color: [number, number, number];
    activeROIId: string | null;
    activeTool: SegmentationModeTool;
    showOnlyActive: boolean;
    aiPreviewMask: unknown | null;
    isPinned: boolean;
    activeClassId: number;
    tool: SegmentationTool;
    brushRadius: number;
    paintValue: 0 | 1;
    thresholdMin: number;
    thresholdMax: number;
    regionGrowTolerance: number;
    regionGrowSliceRadius: number;
}

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
