export interface ThresholdTaskRequest {
    id: number;
    type: 'threshold-slice';
    width: number;
    height: number;
    min: number;
    max: number;
    values: ArrayBuffer;
}

export interface RegionGrowTaskRequest {
    id: number;
    type: 'region-grow-slice';
    width: number;
    height: number;
    seedIndex: number;
    tolerance: number;
    values: ArrayBuffer;
}

export interface EncodeBinaryMaskRLETaskRequest {
    id: number;
    type: 'encode-binary-mask-rle';
    bits: ArrayBuffer;
}

export interface DecodeBinaryMaskRLETaskRequest {
    id: number;
    type: 'decode-binary-mask-rle';
    totalVoxels: number;
    startsWith: 0 | 1;
    runs: ArrayBuffer;
}

export type WorkerTaskRequest =
    | ThresholdTaskRequest
    | RegionGrowTaskRequest
    | EncodeBinaryMaskRLETaskRequest
    | DecodeBinaryMaskRLETaskRequest;

export interface WorkerIndicesResponse {
    id: number;
    ok: true;
    resultType: 'indices';
    indices: ArrayBufferLike;
}

export interface WorkerBinaryRLEResponse {
    id: number;
    ok: true;
    resultType: 'binary-rle';
    totalVoxels: number;
    oneCount: number;
    startsWith: 0 | 1;
    runs: ArrayBufferLike;
}

export interface WorkerBitsResponse {
    id: number;
    ok: true;
    resultType: 'bits';
    oneCount: number;
    bits: ArrayBufferLike;
}

export interface WorkerErrorResponse {
    id: number;
    ok: false;
    error: string;
}

export type WorkerSuccessResponse =
    | WorkerIndicesResponse
    | WorkerBinaryRLEResponse
    | WorkerBitsResponse;

export type WorkerTaskResponse = WorkerSuccessResponse | WorkerErrorResponse;

