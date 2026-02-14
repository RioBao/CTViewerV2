import { decodeBinaryMaskBitsRLE, encodeBinaryMaskBitsRLE } from '../MaskPersistence.js';
import type {
    DecodeBinaryMaskRLETaskRequest,
    EncodeBinaryMaskRLETaskRequest,
    RegionGrowTaskRequest,
    ThresholdTaskRequest,
    WorkerBinaryRLEResponse,
    WorkerBitsResponse,
    WorkerErrorResponse,
    WorkerIndicesResponse,
    WorkerTaskRequest,
    WorkerTaskResponse,
} from './SegmentationWorkerProtocol.js';

const workerScope = self as unknown as {
    onmessage: ((event: MessageEvent<WorkerTaskRequest>) => void) | null;
    postMessage: (message: WorkerTaskResponse, transfer?: Transferable[]) => void;
};

function runThresholdTask(req: ThresholdTaskRequest): Uint32Array {
    const values = new Float32Array(req.values);
    const out = new Uint32Array(values.length);
    let count = 0;
    for (let i = 0; i < values.length; i++) {
        const v = values[i];
        if (v >= req.min && v <= req.max) {
            out[count++] = i;
        }
    }
    return out.subarray(0, count);
}

function runRegionGrowTask(req: RegionGrowTaskRequest): Uint32Array {
    const values = new Float32Array(req.values);
    const { width, height } = req;
    if (req.seedIndex < 0 || req.seedIndex >= values.length) {
        return new Uint32Array(0);
    }

    const visited = new Uint8Array(values.length);
    const queue = new Uint32Array(values.length);
    let qHead = 0;
    let qTail = 0;

    const selected = new Uint32Array(values.length);
    let selectedCount = 0;

    const seedValue = values[req.seedIndex];
    const tol = Math.max(0, req.tolerance);

    const tryVisit = (idx: number): void => {
        if (idx < 0 || idx >= values.length) return;
        if (visited[idx] !== 0) return;
        visited[idx] = 1;
        if (Math.abs(values[idx] - seedValue) <= tol) {
            queue[qTail++] = idx;
            selected[selectedCount++] = idx;
        }
    };

    tryVisit(req.seedIndex);

    while (qHead < qTail) {
        const idx = queue[qHead++];
        const x = idx % width;
        const y = Math.floor(idx / width);

        for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
                if (dx === 0 && dy === 0) continue;
                const nx = x + dx;
                const ny = y + dy;
                if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
                tryVisit(ny * width + nx);
            }
        }
    }

    return selected.subarray(0, selectedCount);
}

function runEncodeBinaryMaskRLETask(req: EncodeBinaryMaskRLETaskRequest): WorkerBinaryRLEResponse {
    const bits = new Uint8Array(req.bits);
    const encoded = encodeBinaryMaskBitsRLE(bits);
    return {
        id: req.id,
        ok: true,
        resultType: 'binary-rle',
        totalVoxels: encoded.totalVoxels,
        oneCount: encoded.oneCount,
        startsWith: encoded.startsWith,
        runs: encoded.runs.buffer,
    };
}

function runDecodeBinaryMaskRLETask(req: DecodeBinaryMaskRLETaskRequest): WorkerBitsResponse {
    const runs = new Uint32Array(req.runs);
    const bits = decodeBinaryMaskBitsRLE({
        totalVoxels: req.totalVoxels,
        startsWith: req.startsWith,
        runs,
    });
    let oneCount = 0;
    for (let i = 0; i < bits.length; i++) {
        if (bits[i] !== 0) oneCount++;
    }
    return {
        id: req.id,
        ok: true,
        resultType: 'bits',
        oneCount,
        bits: bits.buffer,
    };
}

function runSliceTask(req: ThresholdTaskRequest | RegionGrowTaskRequest): WorkerIndicesResponse {
    const result = req.type === 'threshold-slice'
        ? runThresholdTask(req)
        : runRegionGrowTask(req);
    return {
        id: req.id,
        ok: true,
        resultType: 'indices',
        indices: result.buffer,
    };
}

workerScope.onmessage = (event: MessageEvent<WorkerTaskRequest>) => {
    const req = event.data;
    try {
        switch (req.type) {
            case 'threshold-slice':
            case 'region-grow-slice': {
                const response = runSliceTask(req);
                workerScope.postMessage(response satisfies WorkerTaskResponse, [response.indices as ArrayBuffer]);
                return;
            }
            case 'encode-binary-mask-rle': {
                const response = runEncodeBinaryMaskRLETask(req);
                workerScope.postMessage(response satisfies WorkerTaskResponse, [response.runs as ArrayBuffer]);
                return;
            }
            case 'decode-binary-mask-rle': {
                const response = runDecodeBinaryMaskRLETask(req);
                workerScope.postMessage(response satisfies WorkerTaskResponse, [response.bits as ArrayBuffer]);
                return;
            }
        }
    } catch (error) {
        const response: WorkerErrorResponse = {
            id: req.id,
            ok: false,
            error: error instanceof Error ? error.message : String(error),
        };
        workerScope.postMessage(response satisfies WorkerTaskResponse);
    }
};

export {};
