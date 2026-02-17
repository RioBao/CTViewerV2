import * as ort from 'onnxruntime-web/webgpu';
import type { ViewAxis } from '../types.js';

const SAM2_FULL_INPUT_SIZE = 1024;
const SAM2_PREVIEW_INPUT_SIZE = 512;
const SAM2_MASK_SIZE = 256;
const SAM2_CACHE_LIMIT = 8;
const DEFAULT_MASK_THRESHOLD = 0;
const PREVIEW_MASK_LOGIT_THRESHOLD = 0.12;
const AUTO_WINDOW_SAMPLE_CAP = 16384;
const AUTO_HISTOGRAM_BINS = 256;
const AUTO_P01 = 0.01;
const AUTO_P20 = 0.20;
const AUTO_P50 = 0.50;
const AUTO_P80 = 0.80;
const AUTO_P99 = 0.99;

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
    points?: Sam2PromptPoint[];
    windowMin: number;
    windowMax: number;
    inferenceQuality?: Sam2InferenceQuality;
}

export type Sam2InferenceQuality = 'preview' | 'full';
export interface Sam2PromptPoint {
    x: number;
    y: number;
    label: 0 | 1;
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

function percentileFromSorted(sorted: Float32Array, q: number): number {
    if (sorted.length === 0) return 0;
    const qq = Math.max(0, Math.min(1, q));
    if (sorted.length === 1) return sorted[0];
    const pos = qq * (sorted.length - 1);
    const lo = Math.floor(pos);
    const hi = Math.min(sorted.length - 1, Math.ceil(pos));
    if (lo === hi) return sorted[lo];
    const t = pos - lo;
    return sorted[lo] * (1 - t) + sorted[hi] * t;
}

function computeMultiOtsuThresholds3Class(
    values: Float32Array,
    valueMin: number,
    valueMax: number,
    binCountRaw: number,
): [number, number] | null {
    if (values.length < 3) return null;
    if (!(valueMax > valueMin)) return null;
    const binCount = Math.max(16, Math.floor(binCountRaw));
    const histogram = new Float64Array(binCount);
    const range = valueMax - valueMin;
    const scale = (binCount - 1) / range;
    for (let i = 0; i < values.length; i++) {
        const value = values[i];
        const bin = Math.max(0, Math.min(binCount - 1, Math.floor((value - valueMin) * scale)));
        histogram[bin] += 1;
    }

    const total = values.length;
    const probs = new Float64Array(binCount);
    const omega = new Float64Array(binCount);
    const mu = new Float64Array(binCount);
    let omegaAcc = 0;
    let muAcc = 0;
    for (let i = 0; i < binCount; i++) {
        const p = histogram[i] / total;
        probs[i] = p;
        omegaAcc += p;
        muAcc += p * i;
        omega[i] = omegaAcc;
        mu[i] = muAcc;
    }
    const muTotal = mu[binCount - 1];
    const eps = 1e-8;
    let bestScore = -Infinity;
    let bestT1 = -1;
    let bestT2 = -1;
    for (let t1 = 0; t1 < binCount - 2; t1++) {
        const w0 = omega[t1];
        if (w0 <= eps) continue;
        const m0 = mu[t1] / w0;
        for (let t2 = t1 + 1; t2 < binCount - 1; t2++) {
            const w1 = omega[t2] - omega[t1];
            const w2 = 1 - omega[t2];
            if (w1 <= eps || w2 <= eps) continue;
            const m1 = (mu[t2] - mu[t1]) / w1;
            const m2 = (muTotal - mu[t2]) / w2;
            const score = w0 * (m0 - muTotal) * (m0 - muTotal)
                + w1 * (m1 - muTotal) * (m1 - muTotal)
                + w2 * (m2 - muTotal) * (m2 - muTotal);
            if (score > bestScore) {
                bestScore = score;
                bestT1 = t1;
                bestT2 = t2;
            }
        }
    }

    if (bestT1 < 0 || bestT2 <= bestT1) {
        return null;
    }

    const threshold1 = valueMin + ((bestT1 + 1) / binCount) * range;
    const threshold2 = valueMin + ((bestT2 + 1) / binCount) * range;
    if (!(threshold2 > threshold1)) return null;
    return [threshold1, threshold2];
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
            const promptPoints = this.normalizePromptPoints(request.points, width, height, pointX, pointY, pointLabel);

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
            const { inputs, temporaries } = this.buildDecoderInputs(embedding, width, height, promptPoints);
            let selectedIndices: Uint32Array = new Uint32Array(0);
            let iouScore = 0;
            try {
                const decoderOutputs = await this.decoderSession!.run(inputs);
                const masks = getRequiredTensor(decoderOutputs, 'masks');
                const iouPredictions = getRequiredTensor(decoderOutputs, 'iou_predictions');
                const maskThreshold = qualityUsed === 'preview'
                    ? Math.max(this.maskThreshold, PREVIEW_MASK_LOGIT_THRESHOLD)
                    : this.maskThreshold;
                const decoded = await this.decodeMasksToIndices(masks, iouPredictions, width, height, maskThreshold);
                selectedIndices = new Uint32Array(decoded.selectedIndices);
                if (qualityUsed === 'preview') {
                    selectedIndices = this.filterPreviewSmallIslands(selectedIndices, width, height);
                }
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

    private normalizePromptPoints(
        points: Sam2PromptPoint[] | undefined,
        width: number,
        height: number,
        fallbackX: number,
        fallbackY: number,
        fallbackLabel: 0 | 1,
    ): Sam2PromptPoint[] {
        if (!Array.isArray(points) || points.length === 0) {
            return [{ x: fallbackX, y: fallbackY, label: fallbackLabel }];
        }
        const normalized: Sam2PromptPoint[] = [];
        for (const point of points) {
            if (!point) continue;
            const x = Math.max(0, Math.min(width - 1, Math.floor(point.x)));
            const y = Math.max(0, Math.min(height - 1, Math.floor(point.y)));
            const label: 0 | 1 = point.label === 0 ? 0 : 1;
            normalized.push({ x, y, label });
        }
        if (normalized.length === 0) {
            normalized.push({ x: fallbackX, y: fallbackY, label: fallbackLabel });
        }
        return normalized;
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
        const windows = this.buildAutoPseudoRgbWindows(values, windowMin, windowMax);
        const minR = windows[0][0];
        const minG = windows[1][0];
        const minB = windows[2][0];
        const invR = 1 / Math.max(1e-6, windows[0][1] - windows[0][0]);
        const invG = 1 / Math.max(1e-6, windows[1][1] - windows[1][0]);
        const invB = 1 / Math.max(1e-6, windows[2][1] - windows[2][0]);

        for (let y = 0; y < inputSize; y++) {
            const sy = Math.min(height - 1, Math.floor((y * height) / inputSize));
            for (let x = 0; x < inputSize; x++) {
                const sx = Math.min(width - 1, Math.floor((x * width) / inputSize));
                const srcIndex = sy * width + sx;
                const value = values[srcIndex];
                const red = clamp01((value - minR) * invR);
                const green = clamp01((value - minG) * invG);
                const blue = clamp01((value - minB) * invB);
                const dstIndex = y * inputSize + x;
                out[dstIndex] = (red - IMAGE_MEAN[0]) / IMAGE_STD[0];
                out[planeSize + dstIndex] = (green - IMAGE_MEAN[1]) / IMAGE_STD[1];
                out[2 * planeSize + dstIndex] = (blue - IMAGE_MEAN[2]) / IMAGE_STD[2];
            }
        }
    }

    private buildAutoPseudoRgbWindows(
        values: Float32Array,
        fallbackMinRaw: number,
        fallbackMaxRaw: number,
    ): [[number, number], [number, number], [number, number]] {
        const sampleCount = Math.max(1, Math.min(values.length, AUTO_WINDOW_SAMPLE_CAP));
        const sample = new Float32Array(sampleCount);
        if (values.length <= sampleCount) {
            sample.set(values.subarray(0, sampleCount));
        } else {
            const scale = (values.length - 1) / Math.max(1, sampleCount - 1);
            for (let i = 0; i < sampleCount; i++) {
                sample[i] = values[Math.floor(i * scale)];
            }
        }
        sample.sort();

        const p01 = percentileFromSorted(sample, AUTO_P01);
        const p20 = percentileFromSorted(sample, AUTO_P20);
        const p50 = percentileFromSorted(sample, AUTO_P50);
        const p80 = percentileFromSorted(sample, AUTO_P80);
        const p99 = percentileFromSorted(sample, AUTO_P99);

        const sampleMin = sample[0];
        const sampleMax = sample[sample.length - 1];
        let fallbackMin = Number.isFinite(fallbackMinRaw) ? fallbackMinRaw : sampleMin;
        let fallbackMax = Number.isFinite(fallbackMaxRaw) ? fallbackMaxRaw : sampleMax;
        if (!(fallbackMax > fallbackMin)) {
            fallbackMin = sampleMin;
            fallbackMax = sampleMax;
        }
        if (!(fallbackMax > fallbackMin)) {
            fallbackMin = 0;
            fallbackMax = 1;
        }

        const normalizeWindow = (minRaw: number, maxRaw: number): [number, number] => {
            let min = Number.isFinite(minRaw) ? minRaw : fallbackMin;
            let max = Number.isFinite(maxRaw) ? maxRaw : fallbackMax;
            if (!(max > min)) {
                min = fallbackMin;
                max = fallbackMax;
            }
            if (!(max > min)) {
                max = min + 1;
            }
            return [min, max];
        };

        const otsuThresholds = computeMultiOtsuThresholds3Class(sample, sampleMin, sampleMax, AUTO_HISTOGRAM_BINS);
        if (otsuThresholds) {
            const [t1, t2] = otsuThresholds;
            const overlap = 0.05 * Math.max(1e-6, fallbackMax - fallbackMin);
            return [
                normalizeWindow(fallbackMin, t1 + overlap), // Air / low density
                normalizeWindow(t1 - overlap, t2 + overlap), // Intermediate material
                normalizeWindow(t2 - overlap, fallbackMax), // Dense / metal
            ];
        }

        return [
            normalizeWindow(p01, p80), // Low density / broad context
            normalizeWindow(p20, p80), // Mid-density focus
            normalizeWindow(p50, p99), // High-density emphasis
        ];
    }

    private buildDecoderInputs(
        embedding: Sam2EmbeddingCacheEntry,
        width: number,
        height: number,
        points: Sam2PromptPoint[],
    ): { inputs: Record<string, ort.Tensor>; temporaries: ort.Tensor[] } {
        const scaleX = width > 1 ? (embedding.inputSize - 1) / (width - 1) : 0;
        const scaleY = height > 1 ? (embedding.inputSize - 1) / (height - 1) : 0;
        const promptCount = points.length;
        const pointCoords = new Float32Array((promptCount + 1) * 2);
        const pointLabels = new Float32Array(promptCount + 1);
        for (let i = 0; i < promptCount; i++) {
            const point = points[i];
            pointCoords[i * 2] = point.x * scaleX;
            pointCoords[i * 2 + 1] = point.y * scaleY;
            pointLabels[i] = point.label;
        }
        // Sentinel token expected by SAM decoder prompt format.
        pointCoords[promptCount * 2] = 0;
        pointCoords[promptCount * 2 + 1] = 0;
        pointLabels[promptCount] = -1;
        const maskInput = new Float32Array(SAM2_MASK_SIZE * SAM2_MASK_SIZE);
        const hasMaskInput = new Float32Array([0]);

        const pointCoordsTensor = new ort.Tensor('float32', pointCoords, [1, promptCount + 1, 2]);
        const pointLabelsTensor = new ort.Tensor('float32', pointLabels, [1, promptCount + 1]);
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
        maskThreshold: number,
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
                if (maskLogit > maskThreshold) {
                    out[count++] = dstRow + x;
                }
            }
        }

        return {
            selectedIndices: out.slice(0, count),
            iouScore: Number.isFinite(bestIou) ? bestIou : 0,
        };
    }

    private filterPreviewSmallIslands(indices: Uint32Array, width: number, height: number): Uint32Array {
        if (indices.length === 0 || width <= 0 || height <= 0) return indices;
        const totalPixels = width * height;
        if (totalPixels <= 0) return indices;
        const minComponentSize = Math.max(12, Math.min(96, Math.round(totalPixels * 0.00005)));

        const mask = new Uint8Array(totalPixels);
        for (let i = 0; i < indices.length; i++) {
            const idx = indices[i];
            if (idx >= 0 && idx < totalPixels) {
                mask[idx] = 1;
            }
        }

        const visited = new Uint8Array(totalPixels);
        const queue = new Int32Array(indices.length);
        for (let i = 0; i < indices.length; i++) {
            const start = indices[i];
            if (start < 0 || start >= totalPixels) continue;
            if (mask[start] === 0 || visited[start] !== 0) continue;

            let head = 0;
            let tail = 0;
            queue[tail++] = start;
            visited[start] = 1;

            while (head < tail) {
                const current = queue[head++];
                const x = current % width;

                const left = current - 1;
                if (x > 0 && mask[left] !== 0 && visited[left] === 0) {
                    visited[left] = 1;
                    queue[tail++] = left;
                }

                const right = current + 1;
                if (x + 1 < width && mask[right] !== 0 && visited[right] === 0) {
                    visited[right] = 1;
                    queue[tail++] = right;
                }

                const up = current - width;
                if (up >= 0 && mask[up] !== 0 && visited[up] === 0) {
                    visited[up] = 1;
                    queue[tail++] = up;
                }

                const down = current + width;
                if (down < totalPixels && mask[down] !== 0 && visited[down] === 0) {
                    visited[down] = 1;
                    queue[tail++] = down;
                }
            }

            if (tail < minComponentSize) {
                for (let j = 0; j < tail; j++) {
                    mask[queue[j]] = 0;
                }
            }
        }

        const filtered = new Uint32Array(indices.length);
        let count = 0;
        for (let i = 0; i < indices.length; i++) {
            const idx = indices[i];
            if (idx >= 0 && idx < totalPixels && mask[idx] !== 0) {
                filtered[count++] = idx;
            }
        }
        return filtered.slice(0, count);
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
