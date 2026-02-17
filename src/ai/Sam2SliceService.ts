import * as ort from 'onnxruntime-web/webgpu';
import type { ViewAxis } from '../types.js';

const SAM2_FULL_INPUT_SIZE = 1024;
const SAM2_PREVIEW_INPUT_SIZE = 512;
const SAM2_MASK_SIZE = 256;
const SAM2_CACHE_LIMIT = 8;
const DEFAULT_MASK_THRESHOLD = 0;

const DEFAULT_MODEL_BASE_URL = getViteEnvString('VITE_SAM2_MODEL_BASE_URL') ?? 'https://huggingface.co/SharpAI/sam2-hiera-tiny-onnx/resolve/main';
const DEFAULT_ENCODER_URL = getViteEnvString('VITE_SAM2_ENCODER_URL') ?? `${DEFAULT_MODEL_BASE_URL}/encoder.with_runtime_opt.ort`;
const DEFAULT_DECODER_URL = getViteEnvString('VITE_SAM2_DECODER_URL') ?? `${DEFAULT_MODEL_BASE_URL}/decoder.onnx`;
const IMAGE_MEAN: [number, number, number] = [0.485, 0.456, 0.406];
const IMAGE_STD: [number, number, number] = [0.229, 0.224, 0.225];
const ORT_WASM_ASYNCIFY_MJS_URL = new URL('../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.mjs', import.meta.url).href;
const ORT_WASM_ASYNCIFY_WASM_URL = new URL('../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm', import.meta.url).href;
let isOrtWasmConfigured = false;

interface Sam2EmbeddingCacheEntry {
    key: string;
    inputSize: number;
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
    inferenceQuality?: Sam2InferenceQuality;
}

export type Sam2InferenceQuality = 'preview' | 'full';

export interface Sam2SliceTimings {
    encodeMs: number;
    decodeMs: number;
    totalMs: number;
}

export interface Sam2SliceResult {
    selectedIndices: Uint32Array;
    embeddingCacheHit: boolean;
    iouScore: number;
    qualityUsed: Sam2InferenceQuality;
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
    if (!tensor) return;
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
        // ORT 1.23.x webgpu build uses the asyncify runtime. Provide both module + wasm URLs explicitly so
        // Vite serves binaries from node_modules instead of falling back to index.html.
        mjs: ORT_WASM_ASYNCIFY_MJS_URL,
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
    private encoderInputBuffer: Float32Array | null = null;
    private encoderInputTensor: ort.Tensor | null = null;
    private encoderInputSize = 0;
    private initPromise: Promise<void> | null = null;
    private embeddingCache = new Map<string, Sam2EmbeddingCacheEntry>();
    private usingWasmFallback = false;
    private previewInputSizeSupported: boolean | null = null;

    constructor(options: Sam2SliceServiceOptions = {}) {
        this.encoderModelUrl = options.encoderModelUrl ?? DEFAULT_ENCODER_URL;
        this.decoderModelUrl = options.decoderModelUrl ?? DEFAULT_DECODER_URL;
        this.maskThreshold = Number.isFinite(options.maskThreshold ?? NaN)
            ? Number(options.maskThreshold)
            : DEFAULT_MASK_THRESHOLD;
    }

    isUsingWasmFallback(): boolean {
        return this.usingWasmFallback;
    }

    isInitialized(): boolean {
        return !!(this.encoderSession && this.decoderSession);
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
                    qualityUsed: 'full',
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
            const requestedQuality: Sam2InferenceQuality = request.inferenceQuality === 'preview' ? 'preview' : 'full';
            let qualityUsed: Sam2InferenceQuality = requestedQuality;
            let inputSize = this.resolveInputSize(requestedQuality);

            const encodeStartAt = performance.now();
            let cacheKey = this.buildCacheKey(request, width, height, windowMin, windowMax, inputSize);
            let cached = this.embeddingCache.get(cacheKey);
            let embedding: Sam2EmbeddingCacheEntry;
            let cacheHit = false;
            if (cached) {
                cacheHit = true;
                embedding = cached;
                this.embeddingCache.delete(cacheKey);
                this.embeddingCache.set(cacheKey, cached);
            } else {
                try {
                    embedding = await this.computeEmbedding(cacheKey, request.values, width, height, windowMin, windowMax, inputSize);
                } catch (error) {
                    if (
                        requestedQuality === 'preview'
                        && inputSize !== SAM2_FULL_INPUT_SIZE
                        && this.isLikelyInputShapeError(error)
                    ) {
                        this.previewInputSizeSupported = false;
                        qualityUsed = 'full';
                        inputSize = SAM2_FULL_INPUT_SIZE;
                        cacheKey = this.buildCacheKey(request, width, height, windowMin, windowMax, inputSize);
                        cached = this.embeddingCache.get(cacheKey);
                        if (cached) {
                            cacheHit = true;
                            embedding = cached;
                            this.embeddingCache.delete(cacheKey);
                            this.embeddingCache.set(cacheKey, cached);
                        } else {
                            embedding = await this.computeEmbedding(cacheKey, request.values, width, height, windowMin, windowMax, inputSize);
                        }
                    } else {
                        throw error;
                    }
                }
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
                qualityUsed,
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
        this.releaseSessionsSilently();
        disposeTensor(this.encoderInputTensor);
        this.encoderInputBuffer = null;
        this.encoderInputTensor = null;
        this.encoderInputSize = 0;
        this.initPromise = null;
    }

    private releaseSessionsSilently(): void {
        if (this.encoderSession) {
            try { this.encoderSession.release(); } catch { /* ignore */ }
            this.encoderSession = null;
        }
        if (this.decoderSession) {
            try { this.decoderSession.release(); } catch { /* ignore */ }
            this.decoderSession = null;
        }
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
        await Promise.all([
            assertModelUrlIsBinary(this.encoderModelUrl, 'SAM2 encoder model'),
            assertModelUrlIsBinary(this.decoderModelUrl, 'SAM2 decoder model'),
        ]);

        try {
            // Option 1 hybrid backend: encoder on WebGPU, decoder on WASM.
            this.encoderSession = await ort.InferenceSession.create(this.encoderModelUrl, {
                executionProviders: ['webgpu'],
            });
            this.encoderInputName = this.resolveEncoderInputName();
        } catch (error) {
            this.releaseSessionsSilently();
            const details = error instanceof Error ? error.message : String(error);
            throw new Error(`Failed to create SAM2 encoder WebGPU session from "${this.encoderModelUrl}". ${details}`);
        }

        try {
            this.decoderSession = await ort.InferenceSession.create(this.decoderModelUrl, {
                executionProviders: ['wasm'],
            });
        } catch (error) {
            this.releaseSessionsSilently();
            const details = error instanceof Error ? error.message : String(error);
            throw new Error(`Failed to create SAM2 decoder WASM session from "${this.decoderModelUrl}". ${details}`);
        }

        this.usingWasmFallback = false;
        const ortVersion = (ort.env as unknown as { versions?: { common?: string } }).versions?.common ?? 'unknown';
        console.info(`[SAM2] ORT ${ortVersion} backend mode: encoder=webgpu, decoder=wasm`);
    }

    private buildCacheKey(
        request: Sam2SliceRequest,
        width: number,
        height: number,
        windowMin: number,
        windowMax: number,
        inputSize: number,
    ): string {
        return [
            request.volumeKey,
            request.axis,
            request.sliceIndex,
            `${width}x${height}`,
            `enc=${inputSize}`,
            windowMin.toFixed(4),
            windowMax.toFixed(4),
        ].join('|');
    }

    private resolveInputSize(quality: Sam2InferenceQuality): number {
        if (quality === 'preview' && this.previewInputSizeSupported !== false) {
            return SAM2_PREVIEW_INPUT_SIZE;
        }
        return SAM2_FULL_INPUT_SIZE;
    }

    private isLikelyInputShapeError(error: unknown): boolean {
        const message = error instanceof Error ? error.message : String(error);
        return /shape|dimension|mismatch|invalid input|invalid dimensions|inferred/i.test(message);
    }

    private async computeEmbedding(
        cacheKey: string,
        values: Float32Array,
        width: number,
        height: number,
        windowMin: number,
        windowMax: number,
        inputSize: number,
    ): Promise<Sam2EmbeddingCacheEntry> {
        const { tensor: imageTensor, buffer: inputData } = this.getReusableEncoderInputTensor(inputSize);
        this.prepareEncoderInput(inputData, values, width, height, windowMin, windowMax, inputSize);
        const outputs = await this.encoderSession!.run({ [this.encoderInputName]: imageTensor });
        const imageEmbed = await this.toCpuFloat32Tensor(getRequiredTensor(outputs, 'image_embed'));
        const highResFeat0 = await this.toCpuFloat32Tensor(getRequiredTensor(outputs, 'high_res_feats_0'));
        const highResFeat1 = await this.toCpuFloat32Tensor(getRequiredTensor(outputs, 'high_res_feats_1'));
        return {
            key: cacheKey,
            inputSize,
            imageEmbed,
            highResFeat0,
            highResFeat1,
        };
    }

    private getReusableEncoderInputTensor(inputSize: number): { tensor: ort.Tensor; buffer: Float32Array } {
        if (
            !this.encoderInputBuffer
            || !this.encoderInputTensor
            || this.encoderInputSize !== inputSize
        ) {
            disposeTensor(this.encoderInputTensor);
            this.encoderInputBuffer = new Float32Array(3 * inputSize * inputSize);
            this.encoderInputTensor = new ort.Tensor('float32', this.encoderInputBuffer, [1, 3, inputSize, inputSize]);
            this.encoderInputSize = inputSize;
        }
        return {
            tensor: this.encoderInputTensor,
            buffer: this.encoderInputBuffer,
        };
    }

    private prepareEncoderInput(
        out: Float32Array,
        values: Float32Array,
        width: number,
        height: number,
        windowMin: number,
        windowMax: number,
        inputSize: number,
    ): void {
        const planeSize = inputSize * inputSize;
        const invRange = 1 / Math.max(1e-6, windowMax - windowMin);

        for (let y = 0; y < inputSize; y++) {
            const sy = Math.min(height - 1, Math.floor((y * height) / inputSize));
            for (let x = 0; x < inputSize; x++) {
                const sx = Math.min(width - 1, Math.floor((x * width) / inputSize));
                const srcIndex = sy * width + sx;
                const gray = clamp01((values[srcIndex] - windowMin) * invRange);
                const dstIndex = y * inputSize + x;
                out[dstIndex] = (gray - IMAGE_MEAN[0]) / IMAGE_STD[0];
                out[planeSize + dstIndex] = (gray - IMAGE_MEAN[1]) / IMAGE_STD[1];
                out[2 * planeSize + dstIndex] = (gray - IMAGE_MEAN[2]) / IMAGE_STD[2];
            }
        }
    }

    private buildDecoderInputs(
        embedding: Sam2EmbeddingCacheEntry,
        width: number,
        height: number,
        pointX: number,
        pointY: number,
        pointLabel: 0 | 1,
    ): { inputs: Record<string, ort.Tensor>; temporaries: ort.Tensor[] } {
        const scaleX = width > 1 ? (embedding.inputSize - 1) / (width - 1) : 0;
        const scaleY = height > 1 ? (embedding.inputSize - 1) / (height - 1) : 0;
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
        const maskHeight = Math.max(1, Math.floor(masks.dims[2] ?? SAM2_MASK_SIZE));
        const maskWidth = Math.max(1, Math.floor(masks.dims[3] ?? SAM2_MASK_SIZE));
        const area = maskWidth * maskHeight;
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
            const my = Math.min(maskHeight - 1, Math.floor((y * maskHeight) / height));
            const rowBase = my * maskWidth;
            const dstRow = y * width;
            for (let x = 0; x < width; x++) {
                const mx = Math.min(maskWidth - 1, Math.floor((x * maskWidth) / width));
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
            // Download to CPU and let ORT release the backing GPU resources safely.
            const downloaded = await maybe.getData(true);
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
        console.warn(`[SAM2] WebGPU inference failed; recreating SAM2 sessions and retrying once. ${message}`);
        this.clearEmbeddingCache();
        this.releaseSessionsSilently();
        disposeTensor(this.encoderInputTensor);
        this.encoderInputTensor = null;
        this.encoderInputBuffer = null;
        this.encoderInputSize = 0;
        this.usingWasmFallback = false;
        this.initPromise = null;
        await this.ensureInitialized();
    }
}
