import type {
    WorkerBinaryRLEResponse,
    WorkerBitsResponse,
    WorkerIndicesResponse,
    WorkerSuccessResponse,
    WorkerTaskRequest,
    WorkerTaskResponse,
} from './worker/SegmentationWorkerProtocol.js';

interface ThresholdSliceTask {
    width: number;
    height: number;
    values: Float32Array;
    min: number;
    max: number;
}

interface RegionGrowSliceTask {
    width: number;
    height: number;
    values: Float32Array;
    seedIndex: number;
    tolerance: number;
}

interface EncodeBinaryMaskRLETask {
    bits: Uint8Array;
}

interface DecodeBinaryMaskRLETask {
    totalVoxels: number;
    startsWith: 0 | 1;
    runs: Uint32Array;
}

export interface BinaryMaskRLEWorkerResult {
    totalVoxels: number;
    oneCount: number;
    startsWith: 0 | 1;
    runs: Uint32Array;
}

export interface BinaryMaskBitsWorkerResult {
    oneCount: number;
    bits: Uint8Array;
}

interface PendingTask<T> {
    resolve: (result: unknown) => void;
    reject: (reason: Error) => void;
    parse: (response: WorkerSuccessResponse) => T;
}

/**
 * Small RPC client for segmentation heavy ops (threshold, region-grow).
 */
export class SegmentationWorkerClient {
    private worker: Worker;
    private pending = new Map<number, PendingTask<unknown>>();
    private nextTaskId = 1;

    constructor() {
        this.worker = new Worker(new URL('./worker/SegmentationWorker.ts', import.meta.url), { type: 'module' });
        this.worker.onmessage = (event: MessageEvent<WorkerTaskResponse>) => {
            const response = event.data;
            const task = this.pending.get(response.id);
            if (!task) return;
            this.pending.delete(response.id);
            if (!response.ok) {
                task.reject(new Error(response.error));
                return;
            }
            try {
                task.resolve(task.parse(response));
            } catch (error) {
                task.reject(error instanceof Error ? error : new Error(String(error)));
            }
        };
        this.worker.onerror = (event) => {
            const error = new Error(event.message || 'Segmentation worker failure');
            for (const [, task] of this.pending) {
                task.reject(error);
            }
            this.pending.clear();
        };
    }

    private toTransferArrayBuffer(view: ArrayBufferView): ArrayBuffer {
        if (view.buffer instanceof ArrayBuffer && view.byteOffset === 0 && view.byteLength === view.buffer.byteLength) {
            return view.buffer;
        }
        const out = new Uint8Array(view.byteLength);
        out.set(new Uint8Array(view.buffer, view.byteOffset, view.byteLength));
        return out.buffer;
    }

    private enqueueTask<T>(
        request: WorkerTaskRequest,
        transfer: Transferable[],
        parse: (response: WorkerSuccessResponse) => T,
    ): Promise<T> {
        const id = request.id;
        return new Promise<T>((resolve, reject) => {
            this.pending.set(id, {
                resolve: (value) => resolve(value as T),
                reject,
                parse,
            });
            this.worker.postMessage(request, transfer);
        });
    }

    runThresholdSlice(task: ThresholdSliceTask): Promise<Uint32Array> {
        const id = this.nextTaskId++;
        const valuesBuffer = this.toTransferArrayBuffer(task.values);
        return this.enqueueTask(
            {
                id,
                type: 'threshold-slice',
                width: task.width,
                height: task.height,
                min: task.min,
                max: task.max,
                values: valuesBuffer,
            },
            [valuesBuffer],
            (response): Uint32Array => {
                if (response.resultType !== 'indices') {
                    throw new Error(`Unexpected worker response type: ${response.resultType}`);
                }
                const payload = response as WorkerIndicesResponse;
                return new Uint32Array(payload.indices);
            },
        );
    }

    runRegionGrowSlice(task: RegionGrowSliceTask): Promise<Uint32Array> {
        const id = this.nextTaskId++;
        const valuesBuffer = this.toTransferArrayBuffer(task.values);
        return this.enqueueTask(
            {
                id,
                type: 'region-grow-slice',
                width: task.width,
                height: task.height,
                seedIndex: task.seedIndex,
                tolerance: task.tolerance,
                values: valuesBuffer,
            },
            [valuesBuffer],
            (response): Uint32Array => {
                if (response.resultType !== 'indices') {
                    throw new Error(`Unexpected worker response type: ${response.resultType}`);
                }
                const payload = response as WorkerIndicesResponse;
                return new Uint32Array(payload.indices);
            },
        );
    }

    encodeBinaryMaskRLE(task: EncodeBinaryMaskRLETask): Promise<BinaryMaskRLEWorkerResult> {
        const id = this.nextTaskId++;
        const bitsBuffer = this.toTransferArrayBuffer(task.bits);
        return this.enqueueTask(
            {
                id,
                type: 'encode-binary-mask-rle',
                bits: bitsBuffer,
            },
            [bitsBuffer],
            (response): BinaryMaskRLEWorkerResult => {
                if (response.resultType !== 'binary-rle') {
                    throw new Error(`Unexpected worker response type: ${response.resultType}`);
                }
                const payload = response as WorkerBinaryRLEResponse;
                return {
                    totalVoxels: payload.totalVoxels,
                    oneCount: payload.oneCount,
                    startsWith: payload.startsWith,
                    runs: new Uint32Array(payload.runs),
                };
            },
        );
    }

    decodeBinaryMaskRLE(task: DecodeBinaryMaskRLETask): Promise<BinaryMaskBitsWorkerResult> {
        const id = this.nextTaskId++;
        const runsBuffer = this.toTransferArrayBuffer(task.runs);
        return this.enqueueTask(
            {
                id,
                type: 'decode-binary-mask-rle',
                totalVoxels: task.totalVoxels,
                startsWith: task.startsWith,
                runs: runsBuffer,
            },
            [runsBuffer],
            (response): BinaryMaskBitsWorkerResult => {
                if (response.resultType !== 'bits') {
                    throw new Error(`Unexpected worker response type: ${response.resultType}`);
                }
                const payload = response as WorkerBitsResponse;
                return {
                    oneCount: payload.oneCount,
                    bits: new Uint8Array(payload.bits),
                };
            },
        );
    }

    dispose(): void {
        this.worker.terminate();
        for (const [, task] of this.pending) {
            task.reject(new Error('Segmentation worker disposed'));
        }
        this.pending.clear();
    }
}
