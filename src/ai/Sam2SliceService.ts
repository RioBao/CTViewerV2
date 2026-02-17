import * as ort from 'onnxruntime-web/webgpu';
import type { ViewAxis } from '../types.js';

const SAM2_INPUT_SIZE = 1024;
const SAM2_MASK_SIZE = 256;
const SAM2_CACHE_LIMIT = 8;
const DEFAULT_MASK_THRESHOLD = 0;

const DEFAULT_MODEL_BASE_URL = getViteEnvString('VITE_SAM2_MODEL_BASE_URL') ?? 'https://huggingface.co/SharpAI/sam2-hiera-tiny-onnx/resolve/main';
const DEFAULT_ENCODER_URL = getViteEnvString('VITE_SAM2_ENCODER_URL') ?? `${DEFAULT_MODEL_BASE_URL}/encoder.with_runtime_opt.ort`;
const DEFAULT_DECODER_URL = getViteEnvString('VITE_SAM2_DECODER_URL') ?? `${DEFAULT_MODEL_BASE_URL}/decoder.onnx`;
const IMAGE_MEAN: [number, number, number] = [0.485, 0.456, 0.406];
const IMAGE_STD: [number, number, number] = [0.229, 0.224, 0.225];
const ORT_WASM_ASYNCIFY_WASM_URL = new URL('../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm', import.meta.url).href;
let isOrtWasmConfigured = false;

interface Sam2EmbeddingCacheEntry {
    key: string;
    imageEmbed: ort.Tensor;
    highResFeat0: ort.Tensor;
    highResFeat1: ort.Tensor;
}

export interface Sam2SliceServiceOptions {
    encoderModelUrl?: string;
    decoderModelUrl?: string;
    maskThreshold?: number;
}

export interface Sam2SliceRequest {
    volumeKey: string;
    axis: ViewAxis;
    sliceIndex: number;
    width: number;
    height: number;
    values: Float32Array;
    pointX: number;
    pointY: number;
    pointLabel: 0 | 1;
    windowMin: number;
    windowMax: number;
}

export interface Sam2SliceTimings {
    encodeMs: number;
    decodeMs: number;
    totalMs: number;
}

export interface Sam2SliceResult {
    selectedIndices: Uint32Array;
    embeddingCacheHit: boolean;
    iouScore: number;
    timings: Sam2SliceTimings;
}

function getViteEnvString(key: string): string | null {
    const env = (import.meta as unknown as { env?: Record<string, unknown> }).env;
    const value = env?.[key];
    return typeof value === 'string' && value.trim().length > 0 ? value.trim() : null;
}

async function assertModelUrlIsBinary(url: string, modelLabel: string): Promise<void> {
    try {
        const absoluteUrl = new URL(url, typeof location !== 'undefined' ? location.href : undefined);
        const sameOrigin = typeof location !== 'undefined' && absoluteUrl.origin === location.origin;
        if (!sameOrigin) {
            // Cross-origin hosts (eg. Hugging Face) may reject HEAD even when GET is valid.
            // Let ORT load the model directly and surface a true session-create error if needed.
            return;
        }
    } catch {
        // If URL parsing fails, continue with best-effort check below.
    }

    let response: Response;
    try {
        response = await fetch(url, { method: 'HEAD', cache: 'no-store' });
    } catch (error) {
        const details = error instanceof Error ? error.message : String(error);
        throw new Error(`${modelLabel} is not reachable at "${url}". ${details}`);
    }

    if (!response.ok) {
        throw new Error(`${modelLabel} request failed at "${url}" (HTTP ${response.status}).`);
    }

    const contentType = (response.headers.get('content-type') || '').toLowerCase();
    if (contentType.includes('text/html')) {
        throw new Error(
            `${modelLabel} resolved to HTML at "${url}". This usually means the model file is missing and the dev server returned index.html.`,
        );
    }
    if (contentType.startsWith('text/') || contentType.includes('json')) {
        throw new Error(
            `${modelLabel} at "${url}" has content-type "${contentType}" (text), but a binary ONNX/ORT model file is required.`,
        );
    }

    const contentLengthHeader = response.headers.get('content-length');
    if (contentLengthHeader) {
        const contentLength = Number.parseInt(contentLengthHeader, 10);
        if (Number.isFinite(contentLength) && contentLength > 0 && contentLength < 4096) {
            throw new Error(
                `${modelLabel} at "${url}" is only ${contentLength} bytes; this is too small to be a valid SAM2 model file.`,
            );
        }
    }
}

function clamp01(value: number): number {
    if (!Number.isFinite(value)) return 0;
    return Math.max(0, Math.min(1, value));
}

function disposeTensor(tensor: ort.Tensor | null | undefined): void {
    const disposable = tensor as unknown as { dispose?: () => void };
    if (typeof disposable.dispose === 'function') {
        try {
            disposable.dispose();
        } catch {
            // Ignore runtime-specific dispose failures.
        }
    }
}

function getRequiredTensor(outputs: Record<string, ort.Tensor>, key: string): ort.Tensor {
    const tensor = outputs[key];
    if (!tensor) {
        throw new Error(`SAM2 output missing tensor "${key}"`);
    }
    return tensor;
}

function configureOrtWasmRuntime(): void {
    if (isOrtWasmConfigured) return;

    ort.env.logLevel = 'error';
    ort.env.wasm.wasmPaths = {
        // WebGPU bundle expects the asyncify runtime. Force wasm URL so Vite serves a real binary (not index.html fallback).
        wasm: ORT_WASM_ASYNCIFY_WASM_URL,
    };

    // Most local dev setups are not cross-origin isolated, so force single-threaded wasm fallback.
    if (typeof self !== 'undefined' && !self.crossOriginIsolated) {
        ort.env.wasm.numThreads = 1;
    }

    isOrtWasmConfigured = true;
}

/**
 * Slice-based SAM2 service.
 * - Caches image embeddings per (volume, axis, slice, window)
 * - Runs decoder from click prompt
 * - Returns selected slice indices in source resolution
 */
export class Sam2SliceService {
    private readonly encoderModelUrl: string;
    private readonly decoderModelUrl: string;
    private readonly maskThreshold: number;
    private encoderSession: ort.InferenceSession | null = null;
    private decoderSession: ort.InferenceSession | null = null;
    private encoderInputName = 'input_image';
    private initPromise: Promise<void> | null = null;
    private embeddingCache = new Map<string, Sam2EmbeddingCacheEntry>();

    constructor(options: Sam2SliceServiceOptions = {}) {
        this.encoderModelUrl = options.encoderModelUrl ?? DEFAULT_ENCODER_URL;
        this.decoderModelUrl = options.decoderModelUrl ?? DEFAULT_DECODER_URL;
        this.maskThreshold = Number.isFinite(options.maskThreshold ?? NaN)
            ? Number(options.maskThreshold)
            : DEFAULT_MASK_THRESHOLD;
    }

    isUsingWasmFallback(): boolean {
        return false;
    }

    async segmentFromClick(request: Sam2SliceRequest): Promise<Sam2SliceResult> {
        return this.segmentFromClickInternal(request, true);
    }

    private async segmentFromClickInternal(request: Sam2SliceRequest, allowWebGpuRecoveryRetry: boolean): Promise<Sam2SliceResult> {
        try {
            const startedAt = performance.now();
            await this.ensureInitialized();

            const width = Math.floor(request.width);
            const height = Math.floor(request.height);
            if (width <= 0 || height <= 0) {
                return {
                    selectedIndices: new Uint32Array(0),
                    embeddingCacheHit: false,
                    iouScore: 0,
                    timings: { encodeMs: 0, decodeMs: 0, totalMs: 0 },
                };
            }
            if (request.values.length !== width * height) {
                throw new Error(`SAM2 slice value length mismatch: expected ${width * height}, got ${request.values.length}`);
            }

            const pointX = Math.max(0, Math.min(width - 1, Math.floor(request.pointX)));
            const pointY = Math.max(0, Math.min(height - 1, Math.floor(request.pointY)));
            const pointLabel: 0 | 1 = request.pointLabel === 0 ? 0 : 1;

            const windowMin = Number.isFinite(request.windowMin) ? request.windowMin : 0;
            const windowMaxRaw = Number.isFinite(request.windowMax) ? request.windowMax : windowMin + 1;
            const windowMax = Math.max(windowMin + 1e-6, windowMaxRaw);

            const cacheKey = this.buildCacheKey(request, width, height, windowMin, windowMax);
            const encodeStartAt = performance.now();
            const cached = this.embeddingCache.get(cacheKey);
            let embedding: Sam2EmbeddingCacheEntry;
            let cacheHit = false;
            if (cached) {
                cacheHit = true;
                embedding = cached;
                this.embeddingCache.delete(cacheKey);
                this.embeddingCache.set(cacheKey, cached);
            } else {
                embedding = await this.computeEmbedding(cacheKey, request.values, width, height, windowMin, windowMax);
                this.embeddingCache.set(cacheKey, embedding);
                this.trimCache();
            }
            const encodeMs = performance.now() - encodeStartAt;

            const decodeStartAt = performance.now();
            const { inputs, temporaries } = this.buildDecoderInputs(embedding, width, height, pointX, pointY, pointLabel);
            let selectedIndices = new Uint32Array(0);
            let iouScore = 0;
            try {
                const decoderOutputs = await this.decoderSession!.run(inputs);
                const masks = getRequiredTensor(decoderOutputs, 'masks');
                const iouPredictions = getRequiredTensor(decoderOutputs, 'iou_predictions');
                const decoded = await this.decodeMasksToIndices(masks, iouPredictions, width, height);
                selectedIndices = new Uint32Array(decoded.selectedIndices);
                iouScore = decoded.iouScore;
                disposeTensor(masks);
                disposeTensor(iouPredictions);
            } finally {
                for (const tensor of temporaries) {
                    disposeTensor(tensor);
                }
            }
            const decodeMs = performance.now() - decodeStartAt;

            return {
                selectedIndices,
                embeddingCacheHit: cacheHit,
                iouScore,
                timings: {
                    encodeMs,
                    decodeMs,
                    totalMs: performance.now() - startedAt,
                },
            };
        } catch (error) {
            if (allowWebGpuRecoveryRetry && this.shouldRecoverWebGpu(error)) {
                await this.recoverWebGpuSessions(error);
                return this.segmentFromClickInternal(request, false);
            }
            if (this.shouldRecoverWebGpu(error)) {
                const message = error instanceof Error ? error.message : String(error);
                throw new Error(`SAM2 WebGPU failed after a recovery retry. ${message}`);
            }
            throw error;
        }
    }

    clearEmbeddingCache(): void {
        for (const entry of this.embeddingCache.values()) {
            this.disposeEmbedding(entry);
        }
        this.embeddingCache.clear();
    }

    dispose(): void {
        this.clearEmbeddingCache();
        this.encoderSession = null;
        this.decoderSession = null;
        this.initPromise = null;
    }

    private async ensureInitialized(): Promise<void> {
        if (this.encoderSession && this.decoderSession) return;
        if (!this.initPromise) {
            configureOrtWasmRuntime();
            this.initPromise = this.initSessions();
        }
        await this.initPromise;
    }

    private async initSessions(): Promise<void> {
        const sessionOptions: ort.InferenceSession.SessionOptions = {
            executionProviders: ['webgpu'],
            graphOptimizationLevel: 'disabled',
            enableGraphCapture: false,
        };

        try {
            await Promise.all([
                assertModelUrlIsBinary(this.encoderModelUrl, 'SAM2 encoder model'),
                assertModelUrlIsBinary(this.decoderModelUrl, 'SAM2 decoder model'),
            ]);
            try {
                this.encoderSession = await ort.InferenceSession.create(this.encoderModelUrl, sessionOptions);
                this.encoderInputName = this.resolveEncoderInputName();
            } catch (error) {
                const details = error instanceof Error ? error.message : String(error);
                throw new Error(`Failed to create SAM2 encoder session from "${this.encoderModelUrl}". ${details}`);
            }
            try {
                this.decoderSession = await ort.InferenceSession.create(this.decoderModelUrl, sessionOptions);
            } catch (error) {
                const details = error instanceof Error ? error.message : String(error);
                throw new Error(`Failed to create SAM2 decoder session from "${this.decoderModelUrl}". ${details}`);
            }
        } catch (error) {
            const details = error instanceof Error ? error.message : String(error);
            throw new Error(
                `SAM2 model load failed. Expected binary model files at "${this.encoderModelUrl}" and "${this.decoderModelUrl}". ${details}`,
            );
        }
    }

    private buildCacheKey(
        request: Sam2SliceRequest,
        width: number,
        height: number,
        windowMin: number,
        windowMax: number,
    ): string {
        return [
            request.volumeKey,
            request.axis,
            request.sliceIndex,
            `${width}x${height}`,
            windowMin.toFixed(4),
            windowMax.toFixed(4),
        ].join('|');
    }

    private async computeEmbedding(
        cacheKey: string,
        values: Float32Array,
        width: number,
        height: number,
        windowMin: number,
        windowMax: number,
    ): Promise<Sam2EmbeddingCacheEntry> {
        const inputData = this.prepareEncoderInput(values, width, height, windowMin, windowMax);
        const imageTensor = new ort.Tensor('float32', inputData, [1, 3, SAM2_INPUT_SIZE, SAM2_INPUT_SIZE]);
        try {
            const outputs = await this.encoderSession!.run({ [this.encoderInputName]: imageTensor });
            const imageEmbed = await this.toCpuFloat32Tensor(getRequiredTensor(outputs, 'image_embed'));
            const highResFeat0 = await this.toCpuFloat32Tensor(getRequiredTensor(outputs, 'high_res_feats_0'));
            const highResFeat1 = await this.toCpuFloat32Tensor(getRequiredTensor(outputs, 'high_res_feats_1'));
            return {
                key: cacheKey,
                imageEmbed,
                highResFeat0,
                highResFeat1,
            };
        } finally {
            disposeTensor(imageTensor);
        }
    }

    private prepareEncoderInput(
        values: Float32Array,
        width: number,
        height: number,
        windowMin: number,
        windowMax: number,
    ): Float32Array {
        const out = new Float32Array(3 * SAM2_INPUT_SIZE * SAM2_INPUT_SIZE);
        const planeSize = SAM2_INPUT_SIZE * SAM2_INPUT_SIZE;
        const invRange = 1 / Math.max(1e-6, windowMax - windowMin);

        for (let y = 0; y < SAM2_INPUT_SIZE; y++) {
            const sy = Math.min(height - 1, Math.floor((y * height) / SAM2_INPUT_SIZE));
            for (let x = 0; x < SAM2_INPUT_SIZE; x++) {
                const sx = Math.min(width - 1, Math.floor((x * width) / SAM2_INPUT_SIZE));
                const srcIndex = sy * width + sx;
                const gray = clamp01((values[srcIndex] - windowMin) * invRange);
                const dstIndex = y * SAM2_INPUT_SIZE + x;
                out[dstIndex] = (gray - IMAGE_MEAN[0]) / IMAGE_STD[0];
                out[planeSize + dstIndex] = (gray - IMAGE_MEAN[1]) / IMAGE_STD[1];
                out[2 * planeSize + dstIndex] = (gray - IMAGE_MEAN[2]) / IMAGE_STD[2];
            }
        }
        return out;
    }

    private buildDecoderInputs(
        embedding: Sam2EmbeddingCacheEntry,
        width: number,
        height: number,
        pointX: number,
        pointY: number,
        pointLabel: 0 | 1,
    ): { inputs: Record<string, ort.Tensor>; temporaries: ort.Tensor[] } {
        const scaleX = width > 1 ? (SAM2_INPUT_SIZE - 1) / (width - 1) : 0;
        const scaleY = height > 1 ? (SAM2_INPUT_SIZE - 1) / (height - 1) : 0;
        const pointCoords = new Float32Array([
            pointX * scaleX,
            pointY * scaleY,
            0,
            0,
        ]);
        const pointLabels = new Float32Array([pointLabel, -1]);
        const maskInput = new Float32Array(SAM2_MASK_SIZE * SAM2_MASK_SIZE);
        const hasMaskInput = new Float32Array([0]);

        const pointCoordsTensor = new ort.Tensor('float32', pointCoords, [1, 2, 2]);
        const pointLabelsTensor = new ort.Tensor('float32', pointLabels, [1, 2]);
        const maskInputTensor = new ort.Tensor('float32', maskInput, [1, 1, SAM2_MASK_SIZE, SAM2_MASK_SIZE]);
        const hasMaskInputTensor = new ort.Tensor('float32', hasMaskInput, [1]);

        return {
            inputs: {
                image_embed: embedding.imageEmbed,
                high_res_feats_0: embedding.highResFeat0,
                high_res_feats_1: embedding.highResFeat1,
                point_coords: pointCoordsTensor,
                point_labels: pointLabelsTensor,
                mask_input: maskInputTensor,
                has_mask_input: hasMaskInputTensor,
            },
            temporaries: [pointCoordsTensor, pointLabelsTensor, maskInputTensor, hasMaskInputTensor],
        };
    }

    private async decodeMasksToIndices(
        masks: ort.Tensor,
        iouPredictions: ort.Tensor,
        width: number,
        height: number,
    ): Promise<{ selectedIndices: Uint32Array; iouScore: number }> {
        const masksData = await this.readFloat32TensorData(masks);
        const iouData = await this.readFloat32TensorData(iouPredictions);

        const maskCount = masks.dims[1] ?? 1;
        const area = SAM2_MASK_SIZE * SAM2_MASK_SIZE;
        let bestMaskIndex = 0;
        let bestIou = -Infinity;
        for (let i = 0; i < maskCount; i++) {
            const score = iouData[i] ?? -Infinity;
            if (score > bestIou) {
                bestIou = score;
                bestMaskIndex = i;
            }
        }
        const bestOffset = bestMaskIndex * area;

        const out = new Uint32Array(width * height);
        let count = 0;
        for (let y = 0; y < height; y++) {
            const my = Math.min(SAM2_MASK_SIZE - 1, Math.floor((y * SAM2_MASK_SIZE) / height));
            const rowBase = my * SAM2_MASK_SIZE;
            const dstRow = y * width;
            for (let x = 0; x < width; x++) {
                const mx = Math.min(SAM2_MASK_SIZE - 1, Math.floor((x * SAM2_MASK_SIZE) / width));
                const maskLogit = masksData[bestOffset + rowBase + mx];
                if (maskLogit > this.maskThreshold) {
                    out[count++] = dstRow + x;
                }
            }
        }

        return {
            selectedIndices: out.slice(0, count),
            iouScore: Number.isFinite(bestIou) ? bestIou : 0,
        };
    }

    private trimCache(): void {
        while (this.embeddingCache.size > SAM2_CACHE_LIMIT) {
            const firstKey = this.embeddingCache.keys().next().value;
            if (!firstKey) return;
            const entry = this.embeddingCache.get(firstKey);
            if (entry) {
                this.disposeEmbedding(entry);
            }
            this.embeddingCache.delete(firstKey);
        }
    }

    private disposeEmbedding(entry: Sam2EmbeddingCacheEntry): void {
        disposeTensor(entry.imageEmbed);
        disposeTensor(entry.highResFeat0);
        disposeTensor(entry.highResFeat1);
    }

    private async toCpuFloat32Tensor(tensor: ort.Tensor): Promise<ort.Tensor> {
        const data = await this.readFloat32TensorData(tensor);
        const dims = [...tensor.dims];
        const cpuTensor = new ort.Tensor('float32', new Float32Array(data), dims);
        disposeTensor(tensor);
        return cpuTensor;
    }

    private async readFloat32TensorData(tensor: ort.Tensor): Promise<Float32Array> {
        const maybe = tensor as unknown as {
            data?: unknown;
            getData?: (releaseData?: boolean) => Promise<unknown>;
        };
        if (maybe.data instanceof Float32Array) {
            return maybe.data;
        }
        if (typeof maybe.getData === 'function') {
            const downloaded = await maybe.getData(false);
            if (downloaded instanceof Float32Array) {
                return downloaded;
            }
        }
        throw new Error('SAM2 decoder returned unsupported tensor types');
    }

    private resolveEncoderInputName(): string {
        const session = this.encoderSession as unknown as { inputNames?: string[] } | null;
        const inputNames = session?.inputNames ?? [];
        if (inputNames.includes('input_image')) return 'input_image';
        if (inputNames.includes('image')) return 'image';
        return inputNames[0] ?? 'input_image';
    }

    private shouldRecoverWebGpu(error: unknown): boolean {
        const message = error instanceof Error ? error.message : String(error);
        return /webgpu validation failed|used in submit while destroyed|device lost|failed to call ortrun\(\)|failed to execute 'mapasync' on 'gpubuffer'|dxgi_error_device_hung/i.test(message);
    }

    private async recoverWebGpuSessions(error: unknown): Promise<void> {
        const message = error instanceof Error ? error.message : String(error);
        console.warn(`[SmartRegion] ORT WebGPU failed, recreating SAM2 sessions and retrying once. ${message}`);
        this.clearEmbeddingCache();
        this.encoderSession = null;
        this.decoderSession = null;
        this.initPromise = null;
        await this.ensureInitialized();
    }
}
