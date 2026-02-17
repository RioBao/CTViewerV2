import { WebGPUContext } from '../gpu/WebGPUContext.js';
import { SliceRenderer } from '../gpu/SliceRenderer.js';
import { MIPRenderer, TF_PRESETS, RENDER_MODE } from '../gpu/MIPRenderer.js';
import { UIState } from './UIState.js';
import { FilePicker, type FileGroup } from './FilePicker.js';
import { loadVolume } from '../loaders/VolumeLoader.js';
import type { VolumeData } from '../data/VolumeData.js';
import type { StreamingVolumeData } from '../data/StreamingVolumeData.js';
import { VolumeStore } from '../data/VolumeStore.js';
import { AppMode, type MaskTypedArray, type SegmentationSettings, type ViewAxis } from '../types.js';
import type { MaskVolume } from '../segmentation/MaskTypes.js';
import { hexToRgb01, type ROIEntry } from '../segmentation/ROIModel.js';
import type { SegmentationOp } from '../segmentation/OpsQueue.js';
import { createPaintStrokeOp } from '../segmentation/ops/PaintStrokeOp.js';
import { createApplySliceSelectionOp } from '../segmentation/ops/ApplySliceSelectionOp.js';
import { SegmentationWorkerClient } from '../segmentation/SegmentationWorkerClient.js';
import { SegmentationGpuCompute } from '../segmentation/SegmentationGpuCompute.js';
import { ROIRegistry } from '../segmentation/ROIRegistry.js';
import { SegmentationStore, type SegmentationTilesSnapshot } from '../segmentation/SegmentationStore.js';
import type { BinaryMaskRLEJson } from '../segmentation/MaskPersistence.js';
import { Sam2SliceService, type Sam2InferenceQuality, type Sam2PromptPoint } from '../ai/Sam2SliceService.js';

const DEG = Math.PI / 180;
const AXES: ViewAxis[] = ['xy', 'xz', 'yz'];
type QualityPreset = 'low' | 'medium' | 'high';
const MAX_SEGMENTATION_SLICE_RADIUS = 20;
const ACTIVE_ROI_STATS_DEBOUNCE_MS = 220;
const ACTIVE_ROI_STATS_CHUNK_TARGET_VOXELS = 1_200_000;
const BRICK_SIZE = 8;
const MAX_SAFE_3D_UPLOAD_BYTES = 384 * 1024 * 1024; // 384 MB conservative budget
const MASK_3D_SYNC_IDLE_MS = 48;
const MASK_3D_SYNC_DRAG_MS = 140;
const REGION_GROW_GPU_DISABLE_AFTER_MS = 900;
const ENABLE_REGION_GROW_WEBGPU = true;
const SMART_REGION_STATUS_DEFAULT = 'Click to add SAM2 + prompts. Shift+click adds - prompts.';
const THRESHOLD_AUTO_SAMPLE_CAP = 65_536;
const THRESHOLD_AUTO_HISTOGRAM_BINS = 256;

interface ActiveRoiStatsAccumulator {
    roiId: string;
    classId: number;
    voxels: number;
    sampleCount: number;
    sum: number;
    sumSq: number;
}

interface SliceHit {
    renderer: SliceRenderer;
    sliceIndex: number;
    sliceX: number;
    sliceY: number;
}

interface RoiMaskPayload {
    format: 'viewer-roi-mask-v1';
    name: string;
    colorHex?: string;
    classId: number;
    dimensions: [number, number, number];
    spacing: [number, number, number];
    voxelCount: number;
    mask: BinaryMaskRLEJson;
}

interface SegmentationRoiMeta {
    id: string;
    classId: number;
    name: string;
    colorHex: string;
    visible: boolean;
    locked: boolean;
}

interface SegmentationPackagePayload {
    format: 'viewer-segmentation-package-v1';
    dimensions: [number, number, number];
    spacing: [number, number, number];
    classDataType: 'uint8' | 'uint16';
    chunkSize: number;
    tiles: SegmentationTilesSnapshot['tiles'];
    rois: SegmentationRoiMeta[];
}

interface Mask3DLookupCache {
    srcDims: [number, number, number];
    dstDims: [number, number, number];
    xStart: Int32Array;
    xEnd: Int32Array;
    yStart: Int32Array;
    yEnd: Int32Array;
    zStart: Int32Array;
    zEnd: Int32Array;
    conservative: boolean;
}

type RegionGrowBackend = 'webgpu' | 'worker';

interface RegionGrowSliceResult {
    selected: Uint32Array;
    backend: RegionGrowBackend;
    elapsedMs: number;
}

interface RegionGrowPerfStats {
    slices: number;
    selected: number;
    elapsedMs: number;
}

interface SmartRegionPreview {
    axis: ViewAxis;
    sliceIndex: number;
    sliceRadius: number;
    width: number;
    height: number;
    classId: number;
    volumeKey: string;
    points: Sam2PromptPoint[];
    windowMin: number;
    windowMax: number;
    iouScore: number;
    qualityUsed: Sam2InferenceQuality;
    selectedIndices: Uint32Array;
}

function createRegionGrowPerfStats(): RegionGrowPerfStats {
    return { slices: 0, selected: 0, elapsedMs: 0 };
}

interface OtsuThresholdStats {
    min: number;
    max: number;
    binaryThreshold: number;
    lowerThreshold: number;
    upperThreshold: number;
}

function computeOtsuThresholdStats(
    values: Float32Array,
    sampleCap: number,
    histogramBinsRaw: number,
): OtsuThresholdStats | null {
    if (values.length === 0) return null;
    const sampleCount = Math.max(1, Math.min(values.length, sampleCap));
    const sample = new Float32Array(sampleCount);
    let min = Infinity;
    let max = -Infinity;
    if (values.length <= sampleCount) {
        for (let i = 0; i < sampleCount; i++) {
            const value = values[i];
            sample[i] = value;
            if (value < min) min = value;
            if (value > max) max = value;
        }
    } else {
        const scale = (values.length - 1) / Math.max(1, sampleCount - 1);
        for (let i = 0; i < sampleCount; i++) {
            const value = values[Math.floor(i * scale)];
            sample[i] = value;
            if (value < min) min = value;
            if (value > max) max = value;
        }
    }
    if (!(max > min) || !Number.isFinite(min) || !Number.isFinite(max)) {
        return null;
    }

    const histogramBins = Math.max(16, Math.floor(histogramBinsRaw));
    const histogram = new Float64Array(histogramBins);
    const range = max - min;
    const scale = (histogramBins - 1) / range;
    for (let i = 0; i < sample.length; i++) {
        const value = sample[i];
        const bin = Math.max(0, Math.min(histogramBins - 1, Math.floor((value - min) * scale)));
        histogram[bin] += 1;
    }
    const total = sample.length;

    // Binary Otsu threshold.
    let bestBinaryScore = -Infinity;
    let bestBinaryThreshold = 0;
    let prefixWeight = 0;
    let prefixMean = 0;
    let totalMean = 0;
    for (let i = 0; i < histogramBins; i++) {
        totalMean += i * histogram[i];
    }
    for (let t = 0; t < histogramBins - 1; t++) {
        const count = histogram[t];
        prefixWeight += count;
        prefixMean += t * count;
        const suffixWeight = total - prefixWeight;
        if (prefixWeight <= 0 || suffixWeight <= 0) continue;
        const mean0 = prefixMean / prefixWeight;
        const mean1 = (totalMean - prefixMean) / suffixWeight;
        const score = prefixWeight * suffixWeight * (mean0 - mean1) * (mean0 - mean1);
        if (score > bestBinaryScore) {
            bestBinaryScore = score;
            bestBinaryThreshold = t;
        }
    }

    // Multi-Otsu (3 classes) thresholds.
    const omega = new Float64Array(histogramBins);
    const mu = new Float64Array(histogramBins);
    let omegaAcc = 0;
    let muAcc = 0;
    for (let i = 0; i < histogramBins; i++) {
        const p = histogram[i] / total;
        omegaAcc += p;
        muAcc += p * i;
        omega[i] = omegaAcc;
        mu[i] = muAcc;
    }
    const muTotal = mu[histogramBins - 1];
    const eps = 1e-8;
    let bestMultiScore = -Infinity;
    let bestT1 = -1;
    let bestT2 = -1;
    for (let t1 = 0; t1 < histogramBins - 2; t1++) {
        const w0 = omega[t1];
        if (w0 <= eps) continue;
        const m0 = mu[t1] / w0;
        for (let t2 = t1 + 1; t2 < histogramBins - 1; t2++) {
            const w1 = omega[t2] - omega[t1];
            const w2 = 1 - omega[t2];
            if (w1 <= eps || w2 <= eps) continue;
            const m1 = (mu[t2] - mu[t1]) / w1;
            const m2 = (muTotal - mu[t2]) / w2;
            const score = w0 * (m0 - muTotal) * (m0 - muTotal)
                + w1 * (m1 - muTotal) * (m1 - muTotal)
                + w2 * (m2 - muTotal) * (m2 - muTotal);
            if (score > bestMultiScore) {
                bestMultiScore = score;
                bestT1 = t1;
                bestT2 = t2;
            }
        }
    }

    if (bestT1 < 0 || bestT2 <= bestT1) {
        const span = Math.max(1, Math.floor(histogramBins / 6));
        bestT1 = Math.max(0, bestBinaryThreshold - span);
        bestT2 = Math.min(histogramBins - 1, bestBinaryThreshold + span);
        if (bestT2 <= bestT1) bestT2 = Math.min(histogramBins - 1, bestT1 + 1);
    }

    const thresholdFromBin = (bin: number): number => min + ((bin + 1) / histogramBins) * range;
    const lowerThreshold = thresholdFromBin(bestT1);
    const upperThreshold = thresholdFromBin(bestT2);
    return {
        min,
        max,
        binaryThreshold: thresholdFromBin(bestBinaryThreshold),
        lowerThreshold: Math.min(lowerThreshold, upperThreshold),
        upperThreshold: Math.max(lowerThreshold, upperThreshold),
    };
}

/**
 * Top-level application orchestrator.
 * Manages WebGPU renderers, user interaction, histogram, and crosshairs.
 */
export class ViewerApp {
    private gpu: WebGPUContext | null = null;
    private uiState = new UIState();
    private filePicker: FilePicker | null = null;
    private volumeStore = new VolumeStore();
    /** Tracks the volume that started a 3D resolution change; used to detect stale async results. */
    private resolution3DVolumeToken: object | null = null;
    /** AbortController for all persistent window/document event listeners. */
    private readonly globalAbort = new AbortController();
    private segmentationStore = new SegmentationStore(512);
    private roiRegistry = new ROIRegistry();
    private maskSliceBuffers: Record<ViewAxis, MaskTypedArray | null> = { xy: null, xz: null, yz: null };
    private maskDisplaySliceBuffers: Record<ViewAxis, MaskTypedArray | null> = { xy: null, xz: null, yz: null };
    private maskPreviewSliceBuffers: Record<ViewAxis, MaskTypedArray | null> = { xy: null, xz: null, yz: null };
    private smartRegionPromptLayers: Record<ViewAxis, HTMLDivElement | null> = { xy: null, xz: null, yz: null };
    private maskPaletteData = new Float32Array(256 * 4);
    private statsRefreshTimer: number | null = null;
    private statsRefreshQueued = false;
    private lastStatsRefreshAt = 0;
    private activeRoiStats: ActiveRoiStatsAccumulator | null = null;
    private activeRoiStatsDirty = true;
    private activeRoiStatsRebuilding = false;
    private activeRoiStatsRebuildToken = 0;
    private activeRoiStatsRebuildTimer: number | null = null;
    private mask3DDirty = true;
    private mask3DPaletteDirty = true;
    private mask3DSyncTimer: number | null = null;
    private mask3DSyncWantsRender = false;
    private mask3DLastSyncAt = 0;
    private mask3DLabelBuffer: Uint8Array | null = null;
    private mask3DLookupCache: Mask3DLookupCache | null = null;
    private segmentationWorker = new SegmentationWorkerClient();
    private segmentationGpuCompute: SegmentationGpuCompute | null = null;
    private segmentationGpuComputeFailed = false;
    private smartRegionService: Sam2SliceService | null = null;
    private smartRegionPreviewOnly = true;
    private smartRegionPreview: SmartRegionPreview | null = null;
    private thresholdGuideCacheKey: string | null = null;
    private thresholdGuideCacheValue: OtsuThresholdStats | null = null;
    private actionShortcutOverride: 0 | 1 | null = null;
    private regionGrowPerfTotals: Record<RegionGrowBackend, RegionGrowPerfStats> = {
        webgpu: createRegionGrowPerfStats(),
        worker: createRegionGrowPerfStats(),
    };
    private segmentationWorkerBusy = false;
    private activePaintStrokeId: number | null = null;
    private nextPaintStrokeId = 1;

    // Renderers
    private sliceRenderers: Record<ViewAxis, SliceRenderer | null> = { xy: null, xz: null, yz: null };
    private sliceCanvases: Record<ViewAxis, HTMLCanvasElement | null> = { xy: null, xz: null, yz: null };
    private mipRenderer: MIPRenderer | null = null;

    // Crosshair state (volume voxel coords)
    private crosshairPos = { x: 0, y: 0, z: 0 };
    private crosshairsEnabled = false;

    // Histogram
    private histogramBins: number[] = [];
    private displayWindowMin = 0;
    private displayWindowMax = 255;
    private histDragging: 'min' | 'max' | null = null;

    // ROI selection
    private roiDragging = false;
    private roiAxis: ViewAxis | null = null;
    private roiStartCSS = { x: 0, y: 0 };
    private roiEndCSS = { x: 0, y: 0 };
    private roiOverlay: HTMLDivElement | null = null;

    // View maximize
    private maximizedView: string | null = null;

    // Active 2D view for keyboard slice navigation
    private activeAxis: ViewAxis = 'xy';

    // 2D drag state
    private slice2DDragging = false;
    private slice2DDidMove = false;
    private slice2DLastX = 0;
    private slice2DLastY = 0;
    private slice2DStartX = 0;
    private slice2DStartY = 0;
    private slice2DDragAxis: ViewAxis | null = null;
    private segmentationDragging = false;
    private segmentationDragAxis: ViewAxis | null = null;
    private viewRotationQuarter = 0; // 0..3, clockwise 90deg steps for all views

    // Render scheduling
    private renderPending = false;
    private pendingAxisRenders = new Set<ViewAxis>();
    private axisRenderPending = false;
    private preferred3DQuality: QualityPreset = 'low';
    private interactionQualityActive = false;
    private interactionQualityTimer: number | null = null;
    private overlayState = {
        histogramOpen: false,
        histogramPinned: false,
        aboutOpen: false,
        footerInfoOpen: false,
        threeDPanelOpen: false,
        toolDockDragging: false,
        toolDockStartX: 0,
        toolDockStartY: 0,
        toolDockStartLeft: 16,
        toolDockStartTop: 80,
        histogramDragging: false,
        histogramStartX: 0,
        histogramStartY: 0,
        histogramStartLeft: 0,
        histogramStartTop: 80,
        segmentationDragging: false,
        segmentationStartX: 0,
        segmentationStartY: 0,
        segmentationStartLeft: 0,
        segmentationStartTop: 268,
        sliceDragging: false,
        sliceStartX: 0,
        sliceStartY: 0,
        sliceStartLeft: 16,
        sliceStartTop: 16,
        topOverlayTimer: null as number | null,
        threeDControlsTimer: null as number | null,
    };

    // DOM references
    private dropZoneEl!: HTMLElement;
    private canvas3D!: HTMLCanvasElement;
    private fileNameEl!: HTMLElement;
    private imageInfoEl!: HTMLElement;
    private zoomLevelEl!: HTMLElement;
    private placeholder!: HTMLElement;
    private ct3DView!: HTMLElement;
    private imageWrapper!: HTMLElement;
    private sliceIndicators!: Record<ViewAxis, HTMLElement>;
    private histogramCanvas!: HTMLCanvasElement;
    private pixelInfoEl!: HTMLElement;
    private pixelInfoGroup!: HTMLElement;
    private viewportContainers!: Record<string, HTMLElement>;
    private topOverlay: HTMLElement | null = null;
    private toolDock: HTMLElement | null = null;
    private toolDockGrip: HTMLElement | null = null;
    private dockLogoBtn: HTMLElement | null = null;
    private aboutPopover: HTMLElement | null = null;
    private histogramOverlay: HTMLElement | null = null;
    private histogramGrip: HTMLElement | null = null;
    private histogramToggleBtn: HTMLElement | null = null;
    private histogramPinBtn: HTMLElement | null = null;
    private footerInfoPanel: HTMLElement | null = null;
    private footerInfoGrid: HTMLElement | null = null;
    private segmentationOverlay: HTMLElement | null = null;
    private segmentationGrip: HTMLElement | null = null;
    private segmentationPinBtn: HTMLElement | null = null;
    private segmentationAddRoiBtn: HTMLElement | null = null;
    private segmentationSettingsBtn: HTMLElement | null = null;
    private segAIStatusEl: HTMLElement | null = null;
    private segAIBackendChipEl: HTMLElement | null = null;
    private segAIPreviewToggle: HTMLInputElement | null = null;
    private segAIApplyPreviewBtn: HTMLButtonElement | null = null;
    private segAIClearPreviewBtn: HTMLButtonElement | null = null;
    private segAIClearCacheBtn: HTMLButtonElement | null = null;
    private segRoiList: HTMLElement | null = null;
    private segToolPalette: HTMLElement | null = null;
    private segModeBtn: HTMLElement | null = null;
    private toolModePanel: HTMLElement | null = null;
    private toolModeSegmentation: HTMLElement | null = null;
    private sliceControls: HTMLElement | null = null;
    private sliceGrip: HTMLElement | null = null;
    private viewport3DControls: HTMLElement | null = null;
    private viewport3DChip: HTMLElement | null = null;
    private viewport3DPanel: HTMLElement | null = null;
    private resolutionChipText: HTMLElement | null = null;
    private segRoiImportInput: HTMLInputElement | null = null;
    private segPackageImportInput: HTMLInputElement | null = null;

    private get maskVolume(): MaskVolume | null {
        return this.segmentationStore.maskVolume;
    }

    private get roiEntries(): ROIEntry[] {
        return this.roiRegistry.entries;
    }

    private get volume(): VolumeData | StreamingVolumeData | null {
        return this.volumeStore.volume;
    }

    private set volume(next: VolumeData | StreamingVolumeData | null) {
        this.volumeStore.volume = next;
    }

    private get current3DResolution(): 'low' | 'mid' | 'full' {
        return this.volumeStore.current3DResolution;
    }

    private set current3DResolution(next: 'low' | 'mid' | 'full') {
        this.volumeStore.current3DResolution = next;
    }

    async initialize(): Promise<void> {
        // Grab DOM references
        this.dropZoneEl = document.getElementById('dropZone') as HTMLElement;
        this.canvas3D = document.getElementById('canvas3D') as HTMLCanvasElement;
        this.fileNameEl = document.getElementById('fileName') as HTMLElement;
        this.imageInfoEl = document.getElementById('imageInfo') as HTMLElement;
        this.zoomLevelEl = document.getElementById('zoomLevel') as HTMLElement;
        this.placeholder = document.getElementById('placeholder') as HTMLElement;
        this.ct3DView = document.getElementById('ct3DView') as HTMLElement;
        this.imageWrapper = document.getElementById('imageWrapper') as HTMLElement;
        this.histogramCanvas = document.getElementById('histogramCanvas') as HTMLCanvasElement;
        this.pixelInfoEl = document.getElementById('pixelInfo') as HTMLElement;
        this.pixelInfoGroup = document.getElementById('pixelInfoGroup') as HTMLElement;
        this.topOverlay = document.getElementById('topOverlay');
        this.toolDock = document.getElementById('toolDock');
        this.toolDockGrip = document.getElementById('toolDockGrip');
        this.dockLogoBtn = document.getElementById('dockLogoBtn');
        this.aboutPopover = document.getElementById('aboutPopover');
        this.histogramOverlay = document.getElementById('histogramOverlay');
        this.histogramGrip = document.getElementById('histogramGrip');
        this.histogramToggleBtn = document.getElementById('histogramToggleBtn');
        this.histogramPinBtn = document.getElementById('histogramPinBtn');
        this.footerInfoPanel = document.getElementById('footerInfoPanel');
        this.footerInfoGrid = document.getElementById('footerInfoGrid');
        this.segmentationOverlay = document.getElementById('segmentationOverlay');
        this.segmentationGrip = document.getElementById('segmentationGrip');
        this.segmentationPinBtn = document.getElementById('segmentationPinBtn');
        this.segmentationAddRoiBtn = document.getElementById('segmentationAddRoiBtn');
        this.segmentationSettingsBtn = document.getElementById('segmentationSettingsBtn');
        this.segAIStatusEl = document.getElementById('segAIStatus');
        this.segAIBackendChipEl = document.getElementById('segAIBackendChip');
        this.segAIPreviewToggle = document.getElementById('segAIPreviewToggle') as HTMLInputElement | null;
        this.segAIApplyPreviewBtn = document.getElementById('segAIApplyPreviewBtn') as HTMLButtonElement | null;
        this.segAIClearPreviewBtn = document.getElementById('segAIClearPreviewBtn') as HTMLButtonElement | null;
        this.segAIClearCacheBtn = document.getElementById('segAIClearCacheBtn') as HTMLButtonElement | null;
        this.segRoiList = document.getElementById('segRoiList');
        this.segToolPalette = document.getElementById('segToolPalette');
        this.segModeBtn = document.getElementById('segModeBtn');
        this.toolModePanel = document.getElementById('toolModePanel');
        this.toolModeSegmentation = document.getElementById('toolModeSegmentation');
        this.sliceControls = document.getElementById('sliceControls');
        this.sliceGrip = document.getElementById('sliceGrip');
        this.viewport3DControls = document.getElementById('viewport3DControls');
        this.viewport3DChip = document.getElementById('viewport3DChip');
        this.viewport3DPanel = document.getElementById('viewport3DPanel');
        this.resolutionChipText = document.getElementById('resolutionChipText');
        this.segRoiImportInput = document.getElementById('segRoiImportInput') as HTMLInputElement | null;
        this.segPackageImportInput = document.getElementById('segPackageImportInput') as HTMLInputElement | null;

        this.sliceCanvases.xy = document.getElementById('canvasXY') as HTMLCanvasElement;
        this.sliceCanvases.xz = document.getElementById('canvasXZ') as HTMLCanvasElement;
        this.sliceCanvases.yz = document.getElementById('canvasYZ') as HTMLCanvasElement;

        this.sliceIndicators = {
            xy: document.getElementById('sliceIndicatorXY') as HTMLElement,
            xz: document.getElementById('sliceIndicatorXZ') as HTMLElement,
            yz: document.getElementById('sliceIndicatorYZ') as HTMLElement,
        };

        this.viewportContainers = {
            xy: this.sliceCanvases.xy!.closest('.viewport-container') as HTMLElement,
            xz: this.sliceCanvases.xz!.closest('.viewport-container') as HTMLElement,
            yz: this.sliceCanvases.yz!.closest('.viewport-container') as HTMLElement,
            '3d': this.canvas3D.closest('.viewport-container') as HTMLElement,
        };

        const fileInput = document.getElementById('fileInput') as HTMLInputElement;
        const dropZone = this.dropZoneEl;

        // Initialize WebGPU
        try {
            this.gpu = await WebGPUContext.create();
            this.uiState.update({ gpuAvailable: true });

            for (const axis of AXES) {
                this.sliceRenderers[axis] = new SliceRenderer(this.gpu, this.sliceCanvases[axis]!);
            }
            this.mipRenderer = new MIPRenderer(this.gpu, this.canvas3D);
            try {
                if (ENABLE_REGION_GROW_WEBGPU) {
                    this.segmentationGpuCompute = new SegmentationGpuCompute(this.gpu);
                    this.segmentationGpuComputeFailed = false;
                } else {
                    this.segmentationGpuCompute = null;
                    this.segmentationGpuComputeFailed = true;
                    console.info('Region grow WebGPU backend disabled by default; using worker backend.');
                }
            } catch (error) {
                this.segmentationGpuCompute = null;
                this.segmentationGpuComputeFailed = true;
                console.warn('Segmentation GPU compute initialization failed; falling back to worker.', error);
            }

            this.ct3DView.style.display = 'grid';
            this.placeholder.style.display = 'none';
            this.mipRenderer.clear();
        } catch (err) {
            console.error('WebGPU initialization failed:', err);
            this.uiState.update({ gpuAvailable: false });
            this.showGPUError();
            return;
        }

        // File picker
        this.filePicker = new FilePicker(fileInput, dropZone, (groups) => this.handleFiles(groups));

        // Controls
        this.initControls();
        this.init2DControls();
        this.init3DControls();
        this.initSidebarControls();
        this.initHistogramHandles();
        this.bindOverlayUI();
        this.initResizeObservers();
        this.updateToolModePanel();
        this.updateToolModePanelSide();
        window.addEventListener('beforeunload', () => this.segmentationWorker.dispose(), { signal: this.globalAbort.signal });
    }

    /** Tear down all persistent event listeners and owned resources. */
    dispose(): void {
        this.globalAbort.abort();
        this.smartRegionService?.dispose();
        this.smartRegionService = null;
        this.segmentationGpuCompute?.dispose();
        this.segmentationGpuCompute = null;
        this.segmentationWorker.dispose();
        this.segmentationStore.clearMask();
        this.volumeStore.clear({ disposeStreaming: true, resetResolution: true });
    }

    // ================================================================
    // Rendering
    // ================================================================

    private renderAll(): void {
        if (!this.volume) {
            this.mipRenderer?.clear();
            return;
        }
        for (const axis of AXES) {
            this.renderSlice(axis);
        }
        if (this.mipRenderer) {
            this.apply3DOverlaySettings();
            const seg = this.uiState.state.segmentation;
            if ((this.mask3DDirty || this.mask3DPaletteDirty) && this.isSegmentationMode() && seg.visible) {
                this.schedule3DMaskSync({ render: false });
            }
            this.mipRenderer.render();
        }
    }

    private renderSlices(): void {
        if (!this.volume) return;
        for (const axis of AXES) {
            this.renderSlice(axis);
        }
    }

    private renderSlice(axis: ViewAxis): void {
        const renderer = this.sliceRenderers[axis];
        const volume = this.volume;
        if (!renderer || !volume) return;

        const index = this.uiState.state.slices[axis];
        const slice = volume.getSlice(axis, index);
        const seg = this.uiState.state.segmentation;
        renderer.maskOverlayEnabled = !!this.maskVolume && seg.visible && this.isSegmentationMode();
        renderer.maskOverlayOpacity = seg.overlayOpacity;
        renderer.maskOverlayColor = seg.color;
        if (this.isSegmentationMode()) {
            renderer.setMaskPalette(this.buildMaskPalette());
        }

        let maskSliceData: MaskTypedArray | null = null;
        if (this.maskVolume && this.isSegmentationMode()) {
            const maskSlice = this.maskVolume.getSlice(axis, index, this.maskSliceBuffers[axis] ?? undefined);
            this.maskSliceBuffers[axis] = maskSlice.data;
            maskSliceData = this.filterMaskSliceForDisplay(axis, maskSlice.data);
            maskSliceData = this.applySmartRegionPreviewToMaskSlice(axis, index, maskSliceData);
        }

        renderer.updateSlice(slice, maskSliceData);
        renderer.render();
        this.updateSmartRegionPromptMarkers(axis, index, renderer);

        // Update low-res indicator
        const indicator = document.getElementById(`lowres${axis.toUpperCase()}`) as HTMLElement | null;
        if (indicator) {
            if (slice.isLowRes) {
                indicator.classList.add('visible');
            } else {
                indicator.classList.remove('visible');
            }
        }

        // If streaming and got a low-res fallback, request full-res async
        if (slice.isLowRes && volume.isStreaming) {
            volume.getSliceAsync(axis, index);
            // onSliceReady callback will trigger re-render when data arrives
        }

        // Trigger prefetch for nearby slices
        volume.prefetch(axis, index);
    }

    private filterMaskSliceForDisplay(axis: ViewAxis, maskSlice: MaskTypedArray): MaskTypedArray {
        const seg = this.uiState.state.segmentation;
        const active = this.getActiveRoi();
        const requiresFilter = seg.showOnlyActive && !!active;
        if (!requiresFilter) {
            return maskSlice;
        }

        const allowedClassId = active.classId;

        const ctor = maskSlice instanceof Uint16Array ? Uint16Array : Uint8Array;
        let out = this.maskDisplaySliceBuffers[axis];
        if (!out || out.length !== maskSlice.length || out.constructor !== ctor) {
            out = new ctor(maskSlice.length) as MaskTypedArray;
            this.maskDisplaySliceBuffers[axis] = out;
        }

        for (let i = 0; i < maskSlice.length; i++) {
            const classId = maskSlice[i];
            out[i] = classId === allowedClassId ? classId : 0;
        }
        return out;
    }

    private buildMaskPalette(): Float32Array {
        const seg = this.uiState.state.segmentation;
        const active = this.getActiveRoi();
        const [defaultR, defaultG, defaultB] = seg.color;
        const out = this.maskPaletteData;

        // Default palette: class 0 transparent, unknown classes use fallback color.
        for (let classId = 0; classId < 256; classId++) {
            const base = classId * 4;
            out[base] = defaultR;
            out[base + 1] = defaultG;
            out[base + 2] = defaultB;
            out[base + 3] = classId === 0 ? 0.0 : 1.0;
        }

        for (const roi of this.roiEntries) {
            if (roi.classId < 1 || roi.classId > 255) continue;
            const base = roi.classId * 4;
            const [r, g, b] = hexToRgb01(roi.colorHex);
            out[base] = r;
            out[base + 1] = g;
            out[base + 2] = b;
            let alpha = roi.visible ? 1.0 : 0.0;
            if (seg.showOnlyActive && active && roi.id !== active.id) {
                alpha = 0.0;
            }
            out[base + 3] = alpha;
        }

        return out;
    }

    private mark3DMaskDirty(): void {
        this.mask3DDirty = true;
    }

    private mark3DPaletteDirty(): void {
        this.mask3DPaletteDirty = true;
    }

    private clear3DMaskSyncTimer(): void {
        if (this.mask3DSyncTimer != null) {
            clearTimeout(this.mask3DSyncTimer);
            this.mask3DSyncTimer = null;
        }
    }

    private apply3DOverlaySettings(): void {
        if (!this.mipRenderer) return;
        const seg = this.uiState.state.segmentation;
        const enabled = !!this.maskVolume && this.isSegmentationMode() && seg.visible;
        this.mipRenderer.setMaskOverlay(enabled, seg.overlayOpacity);
    }

    private getMask3DLookup(
        srcDims: [number, number, number],
        dstDims: [number, number, number],
    ): Mask3DLookupCache {
        const cache = this.mask3DLookupCache;
        if (cache
            && cache.srcDims[0] === srcDims[0] && cache.srcDims[1] === srcDims[1] && cache.srcDims[2] === srcDims[2]
            && cache.dstDims[0] === dstDims[0] && cache.dstDims[1] === dstDims[1] && cache.dstDims[2] === dstDims[2]) {
            return cache;
        }

        const [sx, sy, sz] = srcDims;
        const [dx, dy, dz] = dstDims;
        const buildAxisRanges = (srcSize: number, dstSize: number): { start: Int32Array; end: Int32Array } => {
            const start = new Int32Array(dstSize);
            const end = new Int32Array(dstSize);
            for (let i = 0; i < dstSize; i++) {
                let from = Math.floor((i * srcSize) / dstSize);
                let to = Math.floor(((i + 1) * srcSize) / dstSize);
                from = Math.max(0, Math.min(srcSize - 1, from));
                to = Math.max(from + 1, Math.min(srcSize, to));
                start[i] = from;
                end[i] = to;
            }
            return { start, end };
        };

        const x = buildAxisRanges(sx, dx);
        const y = buildAxisRanges(sy, dy);
        const z = buildAxisRanges(sz, dz);

        const next: Mask3DLookupCache = {
            srcDims,
            dstDims,
            xStart: x.start,
            xEnd: x.end,
            yStart: y.start,
            yEnd: y.end,
            zStart: z.start,
            zEnd: z.end,
            conservative: dx < sx || dy < sy || dz < sz,
        };
        this.mask3DLookupCache = next;
        return next;
    }

    private buildMask3DUploadData(targetDims: [number, number, number]): Uint8Array | null {
        const mask = this.maskVolume;
        if (!mask) return null;
        const [dx, dy, dz] = targetDims;
        if (dx <= 0 || dy <= 0 || dz <= 0) return null;

        const count = dx * dy * dz;
        let out = this.mask3DLabelBuffer;
        if (!out || out.length !== count) {
            out = new Uint8Array(count);
            this.mask3DLabelBuffer = out;
        }

        const lookup = this.getMask3DLookup(mask.dimensions, targetDims);
        if (lookup.conservative && mask.backend === 'sparse') {
            out.fill(0);
            const [sxSize, sySize, szSize] = mask.dimensions;
            const rowStride = dx * dy;
            const maxClassId = Math.min(255, mask.maxClassId);
            for (let classId = 1; classId <= maxClassId; classId++) {
                if (mask.getClassVoxelCount(classId) === 0) continue;
                mask.forEachVoxelOfClass(classId, (sx, sy, sz) => {
                    const tx = Math.min(dx - 1, Math.floor((sx * dx) / sxSize));
                    const ty = Math.min(dy - 1, Math.floor((sy * dy) / sySize));
                    const tz = Math.min(dz - 1, Math.floor((sz * dz) / szSize));
                    out[tx + ty * dx + tz * rowStride] = classId;
                });
            }
            return out;
        }

        const classCounts = new Uint32Array(256);
        const touchedClasses = new Uint8Array(256);
        let touchedCount = 0;

        let write = 0;
        for (let z = 0; z < dz; z++) {
            const zStart = lookup.zStart[z];
            const zEnd = lookup.zEnd[z];
            for (let y = 0; y < dy; y++) {
                const yStart = lookup.yStart[y];
                const yEnd = lookup.yEnd[y];
                for (let x = 0; x < dx; x++, write++) {
                    const xStart = lookup.xStart[x];
                    const xEnd = lookup.xEnd[x];

                    let bestClass = 0;
                    let bestCount = 0;

                    if (lookup.conservative) {
                        for (let sz = zStart; sz < zEnd; sz++) {
                            for (let sy = yStart; sy < yEnd; sy++) {
                                for (let sx = xStart; sx < xEnd; sx++) {
                                    const classId = mask.getVoxel(sx, sy, sz);
                                    if (classId <= 0) continue;
                                    const clippedClass = classId > 255 ? 255 : classId;
                                    if (classCounts[clippedClass] === 0) {
                                        touchedClasses[touchedCount++] = clippedClass;
                                    }
                                    const nextCount = classCounts[clippedClass] + 1;
                                    classCounts[clippedClass] = nextCount;
                                    if (nextCount > bestCount) {
                                        bestCount = nextCount;
                                        bestClass = clippedClass;
                                    }
                                }
                            }
                        }
                    } else {
                        const sampleX = xStart;
                        const sampleY = yStart;
                        const sampleZ = zStart;
                        const classId = mask.getVoxel(sampleX, sampleY, sampleZ);
                        bestClass = classId > 255 ? 255 : classId;
                    }

                    out[write] = bestClass;
                    for (let i = 0; i < touchedCount; i++) {
                        classCounts[touchedClasses[i]] = 0;
                    }
                    touchedCount = 0;
                }
            }
        }
        return out;
    }

    private schedule3DMaskSync(options: { immediate?: boolean; render?: boolean } = {}): void {
        if (!this.mipRenderer) return;
        const immediate = options.immediate === true;
        const wantsRender = options.render !== false;
        this.mask3DSyncWantsRender = this.mask3DSyncWantsRender || wantsRender;

        if (immediate) {
            this.clear3DMaskSyncTimer();
            const render = this.mask3DSyncWantsRender;
            this.mask3DSyncWantsRender = false;
            this.sync3DMaskOverlayNow(render);
            return;
        }

        if (this.mask3DSyncTimer != null) {
            return;
        }

        const now = performance.now();
        const minInterval = this.segmentationDragging ? MASK_3D_SYNC_DRAG_MS : MASK_3D_SYNC_IDLE_MS;
        const delay = Math.max(0, minInterval - (now - this.mask3DLastSyncAt));
        this.mask3DSyncTimer = window.setTimeout(() => {
            this.mask3DSyncTimer = null;
            const render = this.mask3DSyncWantsRender;
            this.mask3DSyncWantsRender = false;
            this.sync3DMaskOverlayNow(render);
        }, delay);
    }

    private sync3DMaskOverlayNow(render: boolean): void {
        const mip = this.mipRenderer;
        if (!mip) return;
        this.mask3DLastSyncAt = performance.now();
        this.apply3DOverlaySettings();

        if (this.mask3DPaletteDirty) {
            mip.setMaskPalette(this.buildMaskPalette());
            this.mask3DPaletteDirty = false;
        }

        const targetDims = mip.getVolumeDimensions();
        if (!targetDims) {
            if (render) mip.render();
            return;
        }

        if (!this.maskVolume) {
            if (this.mask3DDirty) {
                mip.clearLabelVolume();
                this.mask3DDirty = false;
            }
            if (render) mip.render();
            return;
        }

        const seg = this.uiState.state.segmentation;
        const overlayEnabled = this.isSegmentationMode() && seg.visible;
        if (overlayEnabled && this.mask3DDirty) {
            if (this.maskVolume.getNonZeroVoxelCount() === 0) {
                mip.clearLabelVolume();
            } else {
                const labels = this.buildMask3DUploadData(targetDims);
                if (labels) {
                    mip.uploadLabelVolume(labels, targetDims);
                } else {
                    mip.clearLabelVolume();
                }
            }
            this.mask3DDirty = false;
        }

        if (render) {
            mip.render();
        }
    }

    private updateToolModePanel(): void {
        if (!this.toolModePanel) return;
        const mode = this.uiState.state.appMode;
        const hasVolume = !!this.volume;
        const showSegToolPalette = hasVolume && mode === AppMode.Segmentation;
        const showToolModePanel = showSegToolPalette;
        this.toolModePanel.classList.toggle('open', showToolModePanel);
        this.toolModePanel.setAttribute('aria-hidden', showToolModePanel ? 'false' : 'true');
        if (this.toolDock) {
            this.toolDock.classList.toggle('seg-tools-open', showSegToolPalette);
        }
        if (this.segToolPalette) {
            this.segToolPalette.classList.toggle('open', showSegToolPalette);
            this.segToolPalette.setAttribute('aria-hidden', showSegToolPalette ? 'false' : 'true');
        }
        if (this.toolModeSegmentation) {
            this.toolModeSegmentation.classList.toggle('active', showToolModePanel);
        }
        this.updateToolModePanelSide();
    }

    private updateToolModePanelSide(): void {
        if (!this.toolDock || !this.toolModePanel) return;
        const containerRect = this.dropZoneEl.getBoundingClientRect();
        const dockRect = this.toolDock.getBoundingClientRect();
        const panelWidth = this.toolModePanel.classList.contains('open')
            ? Math.max(220, this.toolModePanel.getBoundingClientRect().width || 268)
            : 0;
        const segPaletteExtra = (this.segToolPalette && this.segToolPalette.classList.contains('open'))
            ? Math.max(48, this.segToolPalette.getBoundingClientRect().width || 48) + 20
            : 0;
        const requiredWidth = panelWidth + segPaletteExtra;
        const leftInContainer = dockRect.left - containerRect.left;
        const rightSpace = containerRect.width - (leftInContainer + dockRect.width) - 8;
        const leftSpace = leftInContainer - 8;
        const placeLeft = rightSpace < requiredWidth && leftSpace > rightSpace;
        this.toolDock.classList.toggle('tool-dock-right', placeLeft);
    }

    private scheduleRender(): void {
        if (this.renderPending) return;
        this.renderPending = true;
        requestAnimationFrame(() => {
            this.renderPending = false;
            this.renderAll();
        });
    }

    private scheduleSliceRender(): void {
        if (this.renderPending) return;
        this.renderPending = true;
        requestAnimationFrame(() => {
            this.renderPending = false;
            this.renderSlices();
        });
    }

    private scheduleAxisRender(axis: ViewAxis): void {
        this.pendingAxisRenders.add(axis);
        if (this.axisRenderPending) return;
        this.axisRenderPending = true;
        requestAnimationFrame(() => {
            this.axisRenderPending = false;
            const axes = Array.from(this.pendingAxisRenders);
            this.pendingAxisRenders.clear();
            for (const a of axes) {
                this.renderSlice(a);
            }
        });
    }

    // ================================================================
    // App mode + segmentation state
    // ================================================================

    private isSegmentationMode(): boolean {
        return this.uiState.state.appMode === AppMode.Segmentation;
    }

    private isMeasuringMode(): boolean {
        return this.uiState.state.appMode === AppMode.Measuring;
    }

    private findRoiById(id: string | null): ROIEntry | null {
        return this.roiRegistry.findById(id);
    }

    private getActiveRoi(): ROIEntry | null {
        return this.findRoiById(this.uiState.state.segmentation.activeROIId);
    }

    private resetRoiEntries(): void {
        if (this.statsRefreshTimer != null) {
            clearTimeout(this.statsRefreshTimer);
            this.statsRefreshTimer = null;
        }
        this.statsRefreshQueued = false;
        this.lastStatsRefreshAt = 0;
        this.invalidateActiveRoiStats();
        this.roiRegistry.reset();
        if (this.segRoiList) {
            this.segRoiList.innerHTML = '';
        }
        this.updateActiveRoiStatsDisplay(null);
        this.mark3DPaletteDirty();
        this.schedule3DMaskSync({ render: false });
    }

    private addRoiEntry(makeActive = true): ROIEntry | null {
        const roi = this.roiRegistry.add(255);
        if (!roi) {
            console.warn('ROI limit reached: uint8 labelmap supports up to 255 classes.');
            return null;
        }

        if (makeActive) {
            this.setActiveRoi(roi.id);
        } else {
            this.refreshSegmentationOverlayUI();
        }
        this.mark3DPaletteDirty();
        this.schedule3DMaskSync({ render: true });
        return roi;
    }

    private ensureActiveRoiExists(): void {
        if (this.roiEntries.length === 0) {
            this.addRoiEntry(true);
            return;
        }
        const active = this.getActiveRoi();
        if (!active) {
            this.setActiveRoi(this.roiEntries[0].id);
        }
    }

    private setActiveRoi(roiId: string): void {
        const roi = this.findRoiById(roiId);
        if (!roi) return;
        this.clearSmartRegionPreview({ render: true, resetStatus: true });
        this.invalidateActiveRoiStats();
        this.setSegmentationState({
            activeROIId: roi.id,
            activeClassId: roi.classId,
            color: hexToRgb01(roi.colorHex),
        });
        this.scheduleActiveRoiStatsRefresh({ force: true });
    }

    private formatNumeric(value: number, digits = 2): string {
        if (!Number.isFinite(value)) return '-';
        return value.toLocaleString(undefined, {
            minimumFractionDigits: 0,
            maximumFractionDigits: digits,
        });
    }

    private computeAutoRegionGrowTolerance(min: number, max: number): number {
        const lo = Math.min(min, max);
        const hi = Math.max(min, max);
        const range = hi - lo;
        if (!Number.isFinite(range) || range <= 0) return 0;
        const scaled = range * 0.05;
        const minTol = Math.max(1e-6, range * 1e-4);
        return Math.min(64, Math.max(minTol, scaled));
    }

    private getSegmentationInputRange(seg: SegmentationSettings): { min: number; max: number; range: number } {
        const sourceMin = this.volume ? this.volume.min : Math.min(seg.thresholdMin, seg.thresholdMax);
        const sourceMax = this.volume ? this.volume.max : Math.max(seg.thresholdMin, seg.thresholdMax);
        const min = Number.isFinite(sourceMin) ? sourceMin : 0;
        const max = Number.isFinite(sourceMax) ? sourceMax : min;
        const lo = Math.min(min, max);
        const hi = Math.max(min, max);
        return {
            min: lo,
            max: hi,
            range: Math.max(0, hi - lo),
        };
    }

    private getSegmentationNumberStep(range: number): number {
        if (!Number.isFinite(range) || range <= 0) return 0.01;
        const rough = range / 1000;
        const exponent = Math.floor(Math.log10(Math.max(rough, 1e-6)));
        const step = 10 ** exponent;
        return Math.max(1e-6, Math.min(1000, step));
    }

    private formatSegmentationInputValue(value: number, step: number): string {
        if (!Number.isFinite(value)) return '0';
        if (!Number.isFinite(step) || step <= 0) return String(value);
        const decimals = step >= 1 ? 0 : Math.min(6, Math.ceil(-Math.log10(step)));
        return value.toFixed(decimals);
    }

    private applySegmentationNumericInputConfig(
        seg: SegmentationSettings,
        thresholdMinInput: HTMLInputElement | null,
        thresholdMaxInput: HTMLInputElement | null,
        toleranceInput: HTMLInputElement | null,
    ): void {
        const rangeInfo = this.getSegmentationInputRange(seg);
        const thresholdStep = this.getSegmentationNumberStep(rangeInfo.range);
        const toleranceStep = this.getSegmentationNumberStep(rangeInfo.range);
        const toleranceDefault = this.computeAutoRegionGrowTolerance(rangeInfo.min, rangeInfo.max);

        if (thresholdMinInput) {
            thresholdMinInput.min = this.formatSegmentationInputValue(rangeInfo.min, thresholdStep);
            thresholdMinInput.max = this.formatSegmentationInputValue(rangeInfo.max, thresholdStep);
            thresholdMinInput.step = this.formatSegmentationInputValue(thresholdStep, thresholdStep);
            thresholdMinInput.value = this.formatSegmentationInputValue(seg.thresholdMin, thresholdStep);
        }
        if (thresholdMaxInput) {
            thresholdMaxInput.min = this.formatSegmentationInputValue(rangeInfo.min, thresholdStep);
            thresholdMaxInput.max = this.formatSegmentationInputValue(rangeInfo.max, thresholdStep);
            thresholdMaxInput.step = this.formatSegmentationInputValue(thresholdStep, thresholdStep);
            thresholdMaxInput.value = this.formatSegmentationInputValue(seg.thresholdMax, thresholdStep);
        }
        if (toleranceInput) {
            toleranceInput.min = '0';
            toleranceInput.step = this.formatSegmentationInputValue(toleranceStep, toleranceStep);
            toleranceInput.value = this.formatSegmentationInputValue(seg.regionGrowTolerance, toleranceStep);
            toleranceInput.placeholder = this.formatSegmentationInputValue(toleranceDefault, toleranceStep);
        }
    }

    private getThresholdGuideForActiveSlice(): OtsuThresholdStats | null {
        if (!this.volume) return null;
        const axis = this.activeAxis;
        const sliceIndex = this.uiState.state.slices[axis];
        const volumeKey = this.buildSmartRegionVolumeKey(this.volume);
        const cacheKey = `${volumeKey}|${axis}|${sliceIndex}`;
        if (cacheKey === this.thresholdGuideCacheKey) {
            return this.thresholdGuideCacheValue;
        }
        const slice = this.volume.getSlice(axis, sliceIndex);
        const values = slice.data instanceof Float32Array ? slice.data : new Float32Array(slice.data);
        const stats = computeOtsuThresholdStats(values, THRESHOLD_AUTO_SAMPLE_CAP, THRESHOLD_AUTO_HISTOGRAM_BINS);
        this.thresholdGuideCacheKey = cacheKey;
        this.thresholdGuideCacheValue = stats;
        return stats;
    }

    private syncThresholdRangeUI(seg: SegmentationSettings): void {
        const minSlider = document.getElementById('segThresholdRangeMin') as HTMLInputElement | null;
        const maxSlider = document.getElementById('segThresholdRangeMax') as HTMLInputElement | null;
        const selection = document.getElementById('segThresholdRangeSelection') as HTMLDivElement | null;
        const markerOtsu2 = document.getElementById('segThresholdMarkerOtsu2') as HTMLDivElement | null;
        const markerOtsu3A = document.getElementById('segThresholdMarkerOtsu3A') as HTMLDivElement | null;
        const markerOtsu3B = document.getElementById('segThresholdMarkerOtsu3B') as HTMLDivElement | null;
        if (!minSlider || !maxSlider || !selection || !markerOtsu2 || !markerOtsu3A || !markerOtsu3B) {
            return;
        }

        const rangeInfo = this.getSegmentationInputRange(seg);
        const step = this.getSegmentationNumberStep(rangeInfo.range);
        const lo = rangeInfo.min;
        const hi = rangeInfo.max;
        const safeRange = Math.max(1e-6, hi - lo);

        minSlider.min = this.formatSegmentationInputValue(lo, step);
        minSlider.max = this.formatSegmentationInputValue(hi, step);
        minSlider.step = this.formatSegmentationInputValue(step, step);
        minSlider.value = this.formatSegmentationInputValue(seg.thresholdMin, step);

        maxSlider.min = this.formatSegmentationInputValue(lo, step);
        maxSlider.max = this.formatSegmentationInputValue(hi, step);
        maxSlider.step = this.formatSegmentationInputValue(step, step);
        maxSlider.value = this.formatSegmentationInputValue(seg.thresholdMax, step);

        const curMin = Math.min(seg.thresholdMin, seg.thresholdMax);
        const curMax = Math.max(seg.thresholdMin, seg.thresholdMax);
        const leftPct = ((curMin - lo) / safeRange) * 100;
        const widthPct = ((curMax - curMin) / safeRange) * 100;
        selection.style.left = `${Math.max(0, Math.min(100, leftPct))}%`;
        selection.style.width = `${Math.max(0, Math.min(100, widthPct))}%`;

        const stats = this.getThresholdGuideForActiveSlice();
        if (!stats) {
            markerOtsu2.style.display = 'none';
            markerOtsu3A.style.display = 'none';
            markerOtsu3B.style.display = 'none';
            return;
        }

        const marker2Pct = ((stats.binaryThreshold - lo) / safeRange) * 100;
        const marker3APct = ((stats.lowerThreshold - lo) / safeRange) * 100;
        const marker3BPct = ((stats.upperThreshold - lo) / safeRange) * 100;
        markerOtsu2.style.display = '';
        markerOtsu3A.style.display = '';
        markerOtsu3B.style.display = '';
        markerOtsu2.style.left = `${Math.max(0, Math.min(100, marker2Pct))}%`;
        markerOtsu3A.style.left = `${Math.max(0, Math.min(100, marker3APct))}%`;
        markerOtsu3B.style.left = `${Math.max(0, Math.min(100, marker3BPct))}%`;
    }

    private onThresholdSliderInput(changed: 'min' | 'max'): void {
        const minSlider = document.getElementById('segThresholdRangeMin') as HTMLInputElement | null;
        const maxSlider = document.getElementById('segThresholdRangeMax') as HTMLInputElement | null;
        if (!minSlider || !maxSlider) return;
        const step = Math.max(1e-6, parseFloat(minSlider.step) || 1e-6);
        const lo = parseFloat(minSlider.min);
        const hi = parseFloat(minSlider.max);
        const clampedLo = Number.isFinite(lo) ? lo : -Infinity;
        const clampedHi = Number.isFinite(hi) ? hi : Infinity;
        let minValue = parseFloat(minSlider.value);
        let maxValue = parseFloat(maxSlider.value);
        if (!Number.isFinite(minValue)) minValue = 0;
        if (!Number.isFinite(maxValue)) maxValue = minValue;

        if (changed === 'min' && minValue > maxValue - step) {
            maxValue = minValue + step;
        } else if (changed === 'max' && maxValue < minValue + step) {
            minValue = maxValue - step;
        }

        minValue = Math.max(clampedLo, Math.min(clampedHi, minValue));
        maxValue = Math.max(clampedLo, Math.min(clampedHi, maxValue));

        if (maxValue - minValue < step) {
            if (changed === 'min') {
                maxValue = Math.min(clampedHi, minValue + step);
                minValue = Math.max(clampedLo, maxValue - step);
            } else {
                minValue = Math.max(clampedLo, maxValue - step);
                maxValue = Math.min(clampedHi, minValue + step);
            }
        }

        minSlider.value = String(minValue);
        maxSlider.value = String(maxValue);
        this.setSegmentationState({
            thresholdMin: Math.min(minValue, maxValue),
            thresholdMax: Math.max(minValue, maxValue),
        });
    }

    private rgb01ToHex(color: [number, number, number]): string {
        const toHex = (v: number) => Math.round(Math.max(0, Math.min(1, v)) * 255).toString(16).padStart(2, '0');
        return `#${toHex(color[0])}${toHex(color[1])}${toHex(color[2])}`;
    }

    private cancelActiveRoiStatsRebuild(): void {
        this.activeRoiStatsRebuildToken++;
        this.activeRoiStatsRebuilding = false;
        if (this.activeRoiStatsRebuildTimer != null) {
            clearTimeout(this.activeRoiStatsRebuildTimer);
            this.activeRoiStatsRebuildTimer = null;
        }
    }

    private invalidateActiveRoiStats(): void {
        this.cancelActiveRoiStatsRebuild();
        this.activeRoiStats = null;
        this.activeRoiStatsDirty = true;
    }

    private startActiveRoiStatsRebuild(active: ROIEntry): void {
        const mask = this.maskVolume;
        const volume = this.volume;
        if (!mask || !volume) {
            this.invalidateActiveRoiStats();
            this.updateActiveRoiStatsDisplay(null);
            return;
        }

        this.cancelActiveRoiStatsRebuild();

        const classVoxelCount = mask.getClassVoxelCount(active.classId);
        if (classVoxelCount === 0) {
            const empty: ActiveRoiStatsAccumulator = {
                roiId: active.id,
                classId: active.classId,
                voxels: 0,
                sampleCount: 0,
                sum: 0,
                sumSq: 0,
            };
            this.activeRoiStatsRebuilding = false;
            this.activeRoiStats = empty;
            this.activeRoiStatsDirty = false;
            this.applyActiveRoiStatsDisplay(empty);
            return;
        }

        if (mask.backend === 'sparse') {
            const acc: ActiveRoiStatsAccumulator = {
                roiId: active.id,
                classId: active.classId,
                voxels: 0,
                sampleCount: 0,
                sum: 0,
                sumSq: 0,
            };
            this.segmentationStore.forEachVoxelOfClass(acc.classId, (x, y, z) => {
                acc.voxels++;
                const value = volume.getValue(x, y, z);
                if (value == null) return;
                acc.sampleCount++;
                acc.sum += value;
                acc.sumSq += value * value;
            });
            this.activeRoiStatsRebuilding = false;
            this.activeRoiStats = acc;
            this.activeRoiStatsDirty = false;
            this.applyActiveRoiStatsDisplay(acc);
            return;
        }

        const token = ++this.activeRoiStatsRebuildToken;
        this.activeRoiStatsRebuilding = true;

        const [nx, ny, nz] = mask.dimensions;
        const voxelsPerSlice = Math.max(1, nx * ny);
        const slicesPerChunk = Math.max(1, Math.floor(ACTIVE_ROI_STATS_CHUNK_TARGET_VOXELS / voxelsPerSlice));
        const acc: ActiveRoiStatsAccumulator = {
            roiId: active.id,
            classId: active.classId,
            voxels: 0,
            sampleCount: 0,
            sum: 0,
            sumSq: 0,
        };
        let zCursor = 0;

        const processChunk = () => {
            if (token !== this.activeRoiStatsRebuildToken) return;

            const endZ = Math.min(nz, zCursor + slicesPerChunk);
            for (let z = zCursor; z < endZ; z++) {
                for (let y = 0; y < ny; y++) {
                    for (let x = 0; x < nx; x++) {
                        if (mask.getVoxel(x, y, z) !== acc.classId) continue;
                        acc.voxels++;
                        const value = volume.getValue(x, y, z);
                        if (value == null) continue;
                        acc.sampleCount++;
                        acc.sum += value;
                        acc.sumSq += value * value;
                    }
                }
            }
            zCursor = endZ;
            this.applyActiveRoiStatsDisplay(acc);

            if (zCursor < nz) {
                this.activeRoiStatsRebuildTimer = window.setTimeout(processChunk, 0);
                return;
            }

            if (token !== this.activeRoiStatsRebuildToken) return;
            this.activeRoiStatsRebuildTimer = null;
            this.activeRoiStatsRebuilding = false;
            this.activeRoiStats = acc;
            this.activeRoiStatsDirty = false;
            this.applyActiveRoiStatsDisplay(acc);
        };

        processChunk();
    }

    private applyActiveRoiStatsDisplay(acc: ActiveRoiStatsAccumulator | null): void {
        if (!acc || !this.volume) {
            this.updateActiveRoiStatsDisplay(null);
            return;
        }
        const mean = acc.sampleCount > 0 ? acc.sum / acc.sampleCount : 0;
        const variance = acc.sampleCount > 0 ? Math.max(0, acc.sumSq / acc.sampleCount - mean * mean) : 0;
        const voxelVolume = this.volume.spacing[0] * this.volume.spacing[1] * this.volume.spacing[2];
        this.updateActiveRoiStatsDisplay({
            voxels: acc.voxels,
            volume: acc.voxels * voxelVolume,
            mean,
            std: Math.sqrt(variance),
        });
    }

    private applyVoxelChangeToActiveRoiStats(
        x: number,
        y: number,
        z: number,
        previousClassId: number,
        nextClassId: number,
    ): void {
        if (previousClassId === nextClassId) return;
        if (!this.volume || this.activeRoiStatsDirty || !this.activeRoiStats) return;
        const active = this.getActiveRoi();
        if (!active) {
            this.invalidateActiveRoiStats();
            return;
        }
        if (this.activeRoiStats.roiId !== active.id || this.activeRoiStats.classId !== active.classId) {
            this.invalidateActiveRoiStats();
            return;
        }
        if (previousClassId !== active.classId && nextClassId !== active.classId) return;

        const value = this.volume.getValue(x, y, z);
        if (previousClassId === active.classId) {
            this.activeRoiStats.voxels = Math.max(0, this.activeRoiStats.voxels - 1);
            if (value != null) {
                this.activeRoiStats.sampleCount = Math.max(0, this.activeRoiStats.sampleCount - 1);
                this.activeRoiStats.sum -= value;
                this.activeRoiStats.sumSq -= value * value;
            }
        }
        if (nextClassId === active.classId) {
            this.activeRoiStats.voxels += 1;
            if (value != null) {
                this.activeRoiStats.sampleCount += 1;
                this.activeRoiStats.sum += value;
                this.activeRoiStats.sumSq += value * value;
            }
        }
    }

    private updateActiveRoiStatsDisplay(stats: { voxels: number; volume: number; mean: number; std: number } | null): void {
        const voxelsEl = document.getElementById('segStatsVoxels');
        const volumeEl = document.getElementById('segStatsVolume');
        const meanEl = document.getElementById('segStatsMean');
        const stdEl = document.getElementById('segStatsStd');
        if (!voxelsEl || !volumeEl || !meanEl || !stdEl) return;

        if (!stats) {
            voxelsEl.textContent = '-';
            volumeEl.textContent = '-';
            meanEl.textContent = '-';
            stdEl.textContent = '-';
            return;
        }

        voxelsEl.textContent = this.formatNumeric(stats.voxels, 0);
        volumeEl.textContent = `${this.formatNumeric(stats.volume, 3)} mm3`;
        meanEl.textContent = this.formatNumeric(stats.mean, 2);
        stdEl.textContent = this.formatNumeric(stats.std, 2);
    }

    private refreshActiveRoiStatsNow(): void {
        const active = this.getActiveRoi();
        if (!active || !this.maskVolume || !this.volume) {
            this.invalidateActiveRoiStats();
            this.updateActiveRoiStatsDisplay(null);
            return;
        }

        if (!this.activeRoiStatsDirty
            && this.activeRoiStats
            && this.activeRoiStats.roiId === active.id
            && this.activeRoiStats.classId === active.classId) {
            this.applyActiveRoiStatsDisplay(this.activeRoiStats);
            return;
        }

        if (this.activeRoiStatsRebuilding) {
            return;
        }

        this.startActiveRoiStatsRebuild(active);
    }

    private scheduleActiveRoiStatsRefresh(options: { force?: boolean } = {}): void {
        const force = options.force === true;

        if (this.segmentationDragging && !force) {
            this.statsRefreshQueued = true;
            return;
        }
        this.statsRefreshQueued = false;

        if (this.statsRefreshTimer != null) {
            if (!force) {
                return;
            }
            clearTimeout(this.statsRefreshTimer);
            this.statsRefreshTimer = null;
        }

        const elapsed = performance.now() - this.lastStatsRefreshAt;
        const delay = force ? 0 : Math.max(0, ACTIVE_ROI_STATS_DEBOUNCE_MS - elapsed);
        this.statsRefreshTimer = window.setTimeout(() => {
            this.statsRefreshTimer = null;
            this.lastStatsRefreshAt = performance.now();
            this.refreshActiveRoiStatsNow();
            if (this.statsRefreshQueued && !this.segmentationDragging) {
                this.scheduleActiveRoiStatsRefresh();
            }
        }, delay);
    }

    private refreshSegmentationOverlayUI(): void {
        const seg = this.uiState.state.segmentation;
        const segEnableToggle = document.getElementById('segEnableToggle') as HTMLInputElement | null;
        const segVisibleToggle = document.getElementById('segVisibleToggle') as HTMLInputElement | null;
        const showOnlyActiveToggle = document.getElementById('segShowOnlyActiveToggle') as HTMLInputElement | null;
        const opacitySlider = document.getElementById('segOpacitySlider') as HTMLInputElement | null;
        const opacityValue = document.getElementById('segOpacityValue');
        const segBrushSize = document.getElementById('segBrushSize') as HTMLInputElement | null;
        const segBrushSizeValue = document.getElementById('segBrushSizeValue');
        const segThresholdMin = document.getElementById('segThresholdMin') as HTMLInputElement | null;
        const segThresholdMax = document.getElementById('segThresholdMax') as HTMLInputElement | null;
        const segThresholdRangeMin = document.getElementById('segThresholdRangeMin') as HTMLInputElement | null;
        const segThresholdRangeMax = document.getElementById('segThresholdRangeMax') as HTMLInputElement | null;
        const segGrowTolerance = document.getElementById('segGrowTolerance') as HTMLInputElement | null;
        const segSliceRadius = document.getElementById('segSliceRadius') as HTMLInputElement | null;
        const exportBtn = document.getElementById('segExportRoiBtn') as HTMLButtonElement | null;
        const importBtn = document.getElementById('segImportRoiBtn') as HTMLButtonElement | null;
        const exportSegBtn = document.getElementById('segExportSegBtn') as HTMLButtonElement | null;
        const importSegBtn = document.getElementById('segImportSegBtn') as HTMLButtonElement | null;
        const activeNameEl = document.getElementById('segActiveName');
        const activeClassEl = document.getElementById('segActiveClass');
        const activeStateEl = document.getElementById('segActiveState');
        const activeColorEl = document.getElementById('segActiveColor');
        const active = this.getActiveRoi();

        if (segEnableToggle) segEnableToggle.checked = seg.enabled;
        if (segVisibleToggle) segVisibleToggle.checked = seg.visible;
        if (showOnlyActiveToggle) showOnlyActiveToggle.checked = seg.showOnlyActive;
        if (opacitySlider) opacitySlider.value = seg.overlayOpacity.toFixed(2);
        if (opacityValue) opacityValue.textContent = seg.overlayOpacity.toFixed(2);
        if (segBrushSize) segBrushSize.value = String(seg.brushRadius);
        if (segBrushSizeValue) segBrushSizeValue.textContent = String(seg.brushRadius);
        this.applySegmentationNumericInputConfig(seg, segThresholdMin, segThresholdMax, segGrowTolerance);
        this.syncThresholdRangeUI(seg);
        if (segSliceRadius) segSliceRadius.value = String(seg.regionGrowSliceRadius);
        const autoThresholdDisabled = !this.volume;
        if (segThresholdRangeMin) segThresholdRangeMin.disabled = autoThresholdDisabled;
        if (segThresholdRangeMax) segThresholdRangeMax.disabled = autoThresholdDisabled;
        if (this.segmentationPinBtn) {
            this.segmentationPinBtn.classList.toggle('active', seg.isPinned);
        }
        this.updateSegmentationToolRows(seg.activeTool);
        this.updateSegmentationToolButtons(seg.activeTool);
        this.refreshSegmentationActionButtons();
        if (exportBtn) exportBtn.disabled = !active;
        if (importBtn) importBtn.disabled = !this.volume;
        if (exportSegBtn) exportSegBtn.disabled = !this.volume || !this.maskVolume;
        if (importSegBtn) importSegBtn.disabled = !this.volume;

        if (activeNameEl) activeNameEl.textContent = active ? active.name : 'No active ROI';
        if (activeClassEl) activeClassEl.textContent = active ? `Class ${active.classId}` : '-';
        if (activeStateEl) {
            activeStateEl.textContent = active
                ? (active.locked ? 'Locked for editing' : 'Ready to edit')
                : 'Select an ROI to edit.';
        }
        if (activeColorEl) {
            activeColorEl.style.background = active?.colorHex || '#6b7280';
            activeColorEl.style.opacity = active ? '1' : '0.45';
        }

        this.refreshSmartRegionPreviewControls();
        this.renderRoiList();
    }

    private renderRoiList(): void {
        if (!this.segRoiList) return;
        this.segRoiList.innerHTML = '';
        const activeId = this.uiState.state.segmentation.activeROIId;

        if (this.roiEntries.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'seg-roi-empty';
            empty.textContent = 'No ROIs yet';
            this.segRoiList.appendChild(empty);
            return;
        }

        for (const roi of this.roiEntries) {
            const row = document.createElement('div');
            row.className = 'seg-roi-row';
            if (roi.id === activeId) row.classList.add('active');
            if (roi.aiBusy) row.classList.add('busy');
            row.title = `Class ${roi.classId}`;
            row.addEventListener('click', () => this.setActiveRoi(roi.id));

            const colorInput = document.createElement('input');
            colorInput.type = 'color';
            colorInput.className = 'seg-roi-color';
            colorInput.value = roi.colorHex;
            colorInput.addEventListener('click', (e) => e.stopPropagation());
            colorInput.addEventListener('input', () => {
                roi.colorHex = colorInput.value;
                if (this.uiState.state.segmentation.activeROIId === roi.id) {
                    this.setSegmentationState({ color: hexToRgb01(roi.colorHex) });
                } else {
                    this.refreshSegmentationOverlayUI();
                }
                this.scheduleSliceRender();
                this.mark3DPaletteDirty();
                this.schedule3DMaskSync({ render: true });
            });

            const nameInput = document.createElement('input');
            nameInput.type = 'text';
            nameInput.className = 'seg-roi-name';
            nameInput.value = roi.name;
            nameInput.addEventListener('click', (e) => e.stopPropagation());
            nameInput.addEventListener('change', () => {
                const trimmed = nameInput.value.trim();
                roi.name = trimmed || `ROI ${roi.classId}`;
                nameInput.value = roi.name;
            });

            const visibilityBtn = document.createElement('button');
            visibilityBtn.type = 'button';
            visibilityBtn.className = `seg-roi-btn${roi.visible ? '' : ' off'}`;
            visibilityBtn.textContent = roi.visible ? 'Vis' : 'Hide';
            visibilityBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                roi.visible = !roi.visible;
                this.refreshSegmentationOverlayUI();
                this.scheduleSliceRender();
                this.mark3DPaletteDirty();
                this.schedule3DMaskSync({ render: true });
            });

            const lockBtn = document.createElement('button');
            lockBtn.type = 'button';
            lockBtn.className = `seg-roi-btn${roi.locked ? ' locked' : ''}`;
            lockBtn.textContent = roi.locked ? 'Lock' : 'Edit';
            lockBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                roi.locked = !roi.locked;
                this.refreshSegmentationOverlayUI();
            });

            const busyTag = document.createElement('span');
            busyTag.className = 'seg-roi-busy';
            busyTag.textContent = roi.aiBusy ? '...' : '';

            const menu = document.createElement('details');
            menu.className = 'seg-roi-menu';
            menu.addEventListener('click', (e) => e.stopPropagation());

            const menuSummary = document.createElement('summary');
            menuSummary.textContent = '...';

            const menuPopover = document.createElement('div');
            menuPopover.className = 'seg-roi-menu-popover';

            const duplicateBtn = document.createElement('button');
            duplicateBtn.type = 'button';
            duplicateBtn.className = 'seg-roi-menu-btn';
            duplicateBtn.textContent = 'Duplicate ROI';
            duplicateBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                menu.open = false;
                this.duplicateRoiById(roi.id);
            });

            const mergeBtn = document.createElement('button');
            mergeBtn.type = 'button';
            mergeBtn.className = 'seg-roi-menu-btn';
            mergeBtn.textContent = 'Merge Into Active';
            mergeBtn.disabled = !activeId || activeId === roi.id;
            mergeBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                menu.open = false;
                this.mergeRoiIntoActiveById(roi.id);
            });

            const exportBtn = document.createElement('button');
            exportBtn.type = 'button';
            exportBtn.className = 'seg-roi-menu-btn';
            exportBtn.textContent = 'Export ROI';
            exportBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                menu.open = false;
                void this.exportRoiById(roi.id);
            });

            menuPopover.append(duplicateBtn, mergeBtn, exportBtn);
            menu.append(menuSummary, menuPopover);

            row.append(colorInput, nameInput, visibilityBtn, lockBtn, busyTag, menu);
            this.segRoiList.appendChild(row);
        }
    }

    private duplicateRoiById(sourceRoiId: string): void {
        const source = this.findRoiById(sourceRoiId);
        if (!source) return;
        const duplicate = this.addRoiEntry(false);
        if (!duplicate) return;

        duplicate.name = `${source.name} Copy`;
        duplicate.colorHex = source.colorHex;

        let changed = 0;
        if (this.maskVolume) {
            const bits = this.segmentationStore.buildClassMaskBits(source.classId);
            if (bits) {
                changed = this.segmentationStore.applyBinaryMaskBitsToClass(duplicate.classId, bits);
                this.segmentationStore.clearOps();
            }
        }

        this.invalidateActiveRoiStats();
        this.setActiveRoi(duplicate.id);
        this.refreshSegmentationOverlayUI();
        this.scheduleSliceRender();
        this.scheduleActiveRoiStatsRefresh({ force: true });
        this.mark3DPaletteDirty();
        if (changed > 0) {
            this.mark3DMaskDirty();
        }
        this.schedule3DMaskSync({ immediate: true, render: true });
    }

    private mergeRoiIntoActiveById(sourceRoiId: string): void {
        const active = this.getActiveRoi();
        const source = this.findRoiById(sourceRoiId);
        if (!active || !source || source.id === active.id || !this.maskVolume) return;

        this.invalidateActiveRoiStats();
        const changed = this.segmentationStore.remapClass(source.classId, active.classId);
        this.segmentationStore.clearOps();
        this.roiRegistry.removeById(source.id);

        this.refreshSegmentationOverlayUI();
        this.scheduleSliceRender();
        this.scheduleActiveRoiStatsRefresh({ force: true });
        this.mark3DPaletteDirty();
        if (changed > 0) {
            this.mark3DMaskDirty();
        }
        this.schedule3DMaskSync({ immediate: true, render: true });
    }

    private sanitizeFileStem(value: string, fallback: string): string {
        const normalized = value.trim().replace(/[^a-zA-Z0-9._-]+/g, '_').replace(/^_+|_+$/g, '');
        return normalized || fallback;
    }

    private downloadJson(payload: unknown, stem: string): void {
        const blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = `${stem}.json`;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        setTimeout(() => URL.revokeObjectURL(url), 0);
    }

    private normalizeHexColor(value: unknown): string | null {
        if (typeof value !== 'string') return null;
        const trimmed = value.trim();
        const match = /^#?([0-9a-fA-F]{6})$/.exec(trimmed);
        if (!match) return null;
        return `#${match[1].toLowerCase()}`;
    }

    private isRoiMaskPayload(value: unknown): value is RoiMaskPayload {
        if (!value || typeof value !== 'object') return false;
        const payload = value as Partial<RoiMaskPayload>;
        const dims = payload.dimensions;
        const spacing = payload.spacing;
        const mask = payload.mask as Partial<BinaryMaskRLEJson> | undefined;
        return payload.format === 'viewer-roi-mask-v1'
            && typeof payload.name === 'string'
            && Array.isArray(dims) && dims.length === 3
            && Array.isArray(spacing) && spacing.length === 3
            && !!mask
            && mask.encoding === 'binary-rle-v1'
            && typeof mask.totalVoxels === 'number'
            && typeof mask.oneCount === 'number'
            && (mask.startsWith === 0 || mask.startsWith === 1)
            && Array.isArray(mask.runs);
    }

    private isSegmentationPackagePayload(value: unknown): value is SegmentationPackagePayload {
        if (!value || typeof value !== 'object') return false;
        const payload = value as Partial<SegmentationPackagePayload>;
        return payload.format === 'viewer-segmentation-package-v1'
            && Array.isArray(payload.dimensions) && payload.dimensions.length === 3
            && Array.isArray(payload.spacing) && payload.spacing.length === 3
            && (payload.classDataType === 'uint8' || payload.classDataType === 'uint16')
            && typeof payload.chunkSize === 'number'
            && Array.isArray(payload.tiles)
            && Array.isArray(payload.rois);
    }

    private async exportRoiById(roiId: string): Promise<void> {
        const roi = this.findRoiById(roiId);
        if (!roi || !this.maskVolume || !this.volume) return;

        const bits = this.segmentationStore.buildClassMaskBits(roi.classId);
        if (!bits) return;

        this.setActiveRoiBusy(roi.id, true);
        try {
            const encodedMask = await this.segmentationWorker.encodeBinaryMaskRLE({ bits });
            const payload: RoiMaskPayload = {
                format: 'viewer-roi-mask-v1',
                name: roi.name,
                colorHex: roi.colorHex,
                classId: roi.classId,
                dimensions: this.maskVolume.dimensions,
                spacing: this.volume.spacing,
                voxelCount: encodedMask.oneCount,
                mask: {
                    encoding: 'binary-rle-v1',
                    totalVoxels: encodedMask.totalVoxels,
                    oneCount: encodedMask.oneCount,
                    startsWith: encodedMask.startsWith,
                    runs: Array.from(encodedMask.runs),
                },
            };
            this.downloadJson(payload, this.sanitizeFileStem(roi.name, `roi_${roi.classId}`));
        } catch (error) {
            console.error('Failed to export ROI:', error);
            window.alert('Failed to export ROI mask.');
        } finally {
            this.setActiveRoiBusy(roi.id, false);
        }
    }

    private async exportActiveRoi(): Promise<void> {
        const active = this.getActiveRoi();
        if (!active) return;
        await this.exportRoiById(active.id);
    }

    private async importRoiFromFile(file: File): Promise<void> {
        if (!this.volume) {
            window.alert('Load a volume before importing ROI masks.');
            return;
        }
        this.ensureMaskVolumeForCurrentVolume();
        if (!this.maskVolume) return;

        let parsed: unknown;
        try {
            parsed = JSON.parse(await file.text());
        } catch (error) {
            console.error('Failed parsing ROI JSON:', error);
            window.alert('Selected file is not valid JSON.');
            return;
        }
        if (!this.isRoiMaskPayload(parsed)) {
            window.alert('Unsupported ROI file format.');
            return;
        }
        const payload = parsed;
        const [nx, ny, nz] = this.maskVolume.dimensions;
        const [px, py, pz] = payload.dimensions;
        if (nx !== px || ny !== py || nz !== pz) {
            window.alert(`ROI dimensions do not match current volume (${nx}x${ny}x${nz}).`);
            return;
        }

        const roi = this.addRoiEntry(false);
        if (!roi) {
            window.alert('Unable to create ROI. Label class limit may be reached.');
            return;
        }
        roi.name = payload.name.trim() || roi.name;
        const normalizedColor = this.normalizeHexColor(payload.colorHex);
        if (normalizedColor) {
            roi.colorHex = normalizedColor;
        }
        this.setActiveRoiBusy(roi.id, true);

        try {
            const runs = Uint32Array.from(payload.mask.runs.map((n) => Math.max(0, Math.floor(n))));
            const decoded = await this.segmentationWorker.decodeBinaryMaskRLE({
                totalVoxels: payload.mask.totalVoxels,
                startsWith: payload.mask.startsWith,
                runs,
            });
            const changed = this.segmentationStore.applyBinaryMaskBitsToClass(roi.classId, decoded.bits);
            this.segmentationStore.clearOps();
            this.setActiveRoi(roi.id);
            this.invalidateActiveRoiStats();
            this.scheduleSliceRender();
            this.scheduleActiveRoiStatsRefresh({ force: true });
            this.updatePixelInfo();
            this.refreshSegmentationOverlayUI();
            if (changed > 0) {
                this.mark3DMaskDirty();
            }
            this.mark3DPaletteDirty();
            this.schedule3DMaskSync({ immediate: true, render: true });
        } catch (error) {
            console.error('Failed to import ROI:', error);
            this.roiRegistry.removeById(roi.id);
            this.refreshSegmentationOverlayUI();
            window.alert('Failed to import ROI mask.');
        } finally {
            this.setActiveRoiBusy(roi.id, false);
        }
    }

    private exportSegmentationPackage(): void {
        if (!this.volume || !this.maskVolume) return;
        const snapshot = this.segmentationStore.serializeLabelTiles(64);
        if (!snapshot) return;

        const payload: SegmentationPackagePayload = {
            format: 'viewer-segmentation-package-v1',
            dimensions: snapshot.dimensions,
            spacing: this.volume.spacing,
            classDataType: snapshot.classDataType,
            chunkSize: snapshot.chunkSize,
            tiles: snapshot.tiles,
            rois: this.roiEntries.map((roi) => ({
                id: roi.id,
                classId: roi.classId,
                name: roi.name,
                colorHex: roi.colorHex,
                visible: roi.visible,
                locked: roi.locked,
            })),
        };
        const stem = this.sanitizeFileStem(this.fileNameEl.textContent || 'segmentation', 'segmentation');
        this.downloadJson(payload, `${stem}_segmentation`);
    }

    private async importSegmentationPackageFromFile(file: File): Promise<void> {
        if (!this.volume) {
            window.alert('Load a volume before importing segmentation packages.');
            return;
        }
        let parsed: unknown;
        try {
            parsed = JSON.parse(await file.text());
        } catch (error) {
            console.error('Failed parsing segmentation package JSON:', error);
            window.alert('Selected file is not valid JSON.');
            return;
        }
        if (!this.isSegmentationPackagePayload(parsed)) {
            window.alert('Unsupported segmentation package format.');
            return;
        }
        const payload = parsed;
        const [vx, vy, vz] = this.volume.dimensions;
        const [px, py, pz] = payload.dimensions;
        if (vx !== px || vy !== py || vz !== pz) {
            window.alert(`Package dimensions do not match current volume (${vx}x${vy}x${vz}).`);
            return;
        }

        const recreated = this.segmentationStore.ensureMaskVolume(this.volume.dimensions, {
            preferredBackend: 'auto',
            classDataType: payload.classDataType,
        });
        if (recreated) {
            this.maskSliceBuffers = { xy: null, xz: null, yz: null };
            this.maskDisplaySliceBuffers = { xy: null, xz: null, yz: null };
            this.mask3DLookupCache = null;
            this.mask3DLabelBuffer = null;
            this.invalidateActiveRoiStats();
        }

        try {
            const snapshot: SegmentationTilesSnapshot = {
                format: 'viewer-segmentation-tiles-v1',
                dimensions: payload.dimensions,
                classDataType: payload.classDataType,
                chunkSize: payload.chunkSize,
                tiles: payload.tiles,
            };
            this.segmentationStore.restoreLabelTiles(snapshot, { clearFirst: true });
        } catch (error) {
            console.error('Failed restoring segmentation tiles:', error);
            window.alert('Failed to restore segmentation tiles from file.');
            return;
        }

        this.resetRoiEntries();
        const roiCandidates = payload.rois.filter((roi) =>
            roi
            && typeof roi.id === 'string'
            && typeof roi.classId === 'number'
            && Number.isFinite(roi.classId)
            && roi.classId > 0,
        );
        const sortedRois = [...roiCandidates].sort((a, b) => a.classId - b.classId);
        for (const roi of sortedRois) {
            const color = this.normalizeHexColor(roi.colorHex);
            this.roiRegistry.addWithClassId(Math.floor(roi.classId), {
                id: roi.id,
                name: typeof roi.name === 'string' ? roi.name : undefined,
                colorHex: color ?? undefined,
                visible: typeof roi.visible === 'boolean' ? roi.visible : undefined,
                locked: typeof roi.locked === 'boolean' ? roi.locked : undefined,
            });
        }
        if (this.roiEntries.length === 0) {
            const usedClassIds = this.segmentationStore.collectUsedClassIds();
            for (const classId of usedClassIds) {
                this.roiRegistry.addWithClassId(classId);
            }
        }

        if (this.roiEntries.length > 0) {
            this.setActiveRoi(this.roiEntries[0].id);
        } else {
            this.setSegmentationState({
                activeROIId: null,
                activeClassId: 1,
            });
        }
        this.scheduleSliceRender();
        this.scheduleActiveRoiStatsRefresh({ force: true });
        this.updatePixelInfo();
        this.refreshSegmentationOverlayUI();
        this.mark3DMaskDirty();
        this.mark3DPaletteDirty();
        this.schedule3DMaskSync({ immediate: true, render: true });
    }

    private setAppMode(mode: AppMode): void {
        const current = this.uiState.state.appMode;
        if (current === mode) return;

        if (current === AppMode.Measuring && mode !== AppMode.Measuring) {
            this.roiDragging = false;
            this.removeRoiOverlay();
        }

        if (current === AppMode.Segmentation && mode !== AppMode.Segmentation) {
            this.segmentationDragging = false;
            this.segmentationDragAxis = null;
            this.activePaintStrokeId = null;
            this.clearSmartRegionPreview({ render: true, resetStatus: true });
        }

        this.uiState.update({ appMode: mode });

        if (mode === AppMode.Segmentation) {
            this.ensureMaskVolumeForCurrentVolume();
            if (!this.uiState.state.segmentation.enabled) {
                this.setSegmentationState({ enabled: true });
            }
            this.ensureActiveRoiExists();
        }

        this.syncModeButtons();
        this.updateSegmentationPanelVisibility();
        this.updateInteractionCursor();
        this.refreshFooterInfoDetails();
        this.scheduleActiveRoiStatsRefresh({ force: true });
        this.scheduleSliceRender();
        this.mark3DPaletteDirty();
        this.schedule3DMaskSync({ immediate: true, render: true });
    }

    private toggleSegmentationMode(): void {
        this.setAppMode(this.isSegmentationMode() ? AppMode.Viewing : AppMode.Segmentation);
    }

    private syncModeButtons(): void {
        const mode = this.uiState.state.appMode;
        const roiBtn = document.getElementById('roiBtn');
        if (roiBtn) {
            roiBtn.classList.toggle('active', mode === AppMode.Measuring);
        }
        if (this.segModeBtn) {
            this.segModeBtn.classList.toggle('active', mode === AppMode.Segmentation);
        }
        this.updateToolModePanel();
    }

    private updateSegmentationPanelVisibility(): void {
        const shouldShow = this.shouldKeepSegmentationPanelOpen();
        if (this.segmentationOverlay) {
            this.segmentationOverlay.classList.toggle('open', shouldShow);
            this.segmentationOverlay.setAttribute('aria-hidden', shouldShow ? 'false' : 'true');
            if (shouldShow) {
                this.clampSegmentationOverlayPosition();
            }
        }
        this.refreshSegmentationOverlayUI();
    }

    private shouldKeepSegmentationPanelOpen(): boolean {
        const seg = this.uiState.state.segmentation;
        return this.isSegmentationMode() || seg.isPinned;
    }

    private updateInteractionCursor(): void {
        const useCrosshair = this.isSegmentationMode() || this.isMeasuringMode();
        for (const axis of AXES) {
            const canvas = this.sliceCanvases[axis];
            if (canvas) {
                canvas.style.cursor = useCrosshair ? 'crosshair' : '';
            }
        }
    }

    private getActiveSegmentationModeTool(seg: SegmentationSettings): SegmentationSettings['activeTool'] {
        if (seg.tool === 'brush') {
            return 'brush';
        }
        if (seg.tool === 'threshold') {
            return 'threshold';
        }
        if (seg.tool === 'region-grow') {
            return 'region-grow';
        }
        return 'smart-region';
    }

    private setSegmentationState(patch: Partial<SegmentationSettings>): void {
        const prev = this.uiState.state.segmentation;
        const next: SegmentationSettings = {
            ...prev,
            ...patch,
        };

        if (patch.activeTool !== undefined) {
            switch (patch.activeTool) {
                case 'brush':
                    next.tool = 'brush';
                    break;
                case 'threshold':
                    next.tool = 'threshold';
                    break;
                case 'region-grow':
                    next.tool = 'region-grow';
                    break;
                case 'smart-region':
                    next.tool = 'smart-region';
                    break;
                default:
                    break;
            }
        }

        if (patch.activeROIId !== undefined) {
            const roi = this.findRoiById(patch.activeROIId);
            if (roi) {
                next.activeROIId = roi.id;
                next.activeClassId = roi.classId;
                if (patch.color === undefined) {
                    next.color = hexToRgb01(roi.colorHex);
                }
            } else {
                next.activeROIId = null;
            }
        }
        if (patch.activeClassId !== undefined && patch.activeROIId === undefined) {
            const roi = this.roiRegistry.findByClassId(next.activeClassId);
            if (roi) {
                next.activeROIId = roi.id;
                if (patch.color === undefined) {
                    next.color = hexToRgb01(roi.colorHex);
                }
            }
        }

        if (patch.color !== undefined && next.activeROIId) {
            const active = this.findRoiById(next.activeROIId);
            if (active) {
                active.colorHex = this.rgb01ToHex(patch.color);
            }
        }

        next.overlayOpacity = Math.max(0, Math.min(1, next.overlayOpacity));
        next.brushRadius = Math.max(1, Math.min(64, Math.round(next.brushRadius)));
        next.activeClassId = Math.max(1, Math.min(255, Math.round(next.activeClassId)));
        next.thresholdMin = Number.isFinite(next.thresholdMin) ? next.thresholdMin : 0;
        next.thresholdMax = Number.isFinite(next.thresholdMax) ? next.thresholdMax : 0;
        next.regionGrowTolerance = Number.isFinite(next.regionGrowTolerance) ? Math.max(0, next.regionGrowTolerance) : 0;
        next.regionGrowSliceRadius = Math.max(0, Math.min(MAX_SEGMENTATION_SLICE_RADIUS, Math.round(next.regionGrowSliceRadius)));
        if (this.volume) {
            const lo = Math.min(this.volume.min, this.volume.max);
            const hi = Math.max(this.volume.min, this.volume.max);
            next.thresholdMin = Math.max(lo, Math.min(hi, next.thresholdMin));
            next.thresholdMax = Math.max(lo, Math.min(hi, next.thresholdMax));
        }
        if (patch.activeTool !== undefined) {
            next.activeTool = patch.activeTool;
        } else if (patch.tool !== undefined || patch.paintValue !== undefined) {
            next.activeTool = this.getActiveSegmentationModeTool(next);
        }

        const shouldClearSmartPreview = !!this.smartRegionPreview
            && (
                prev.activeROIId !== next.activeROIId
                || prev.tool !== next.tool
                || prev.activeTool !== next.activeTool
                || !this.isSegmentationMode()
            );
        if (shouldClearSmartPreview) {
            this.clearSmartRegionPreview({ render: true, resetStatus: true });
        }

        this.uiState.update({ segmentation: next });
        if (patch.tool !== undefined || patch.paintValue !== undefined || patch.activeTool !== undefined) {
            this.updateSegmentationToolRows(next.activeTool);
            this.updateSegmentationToolButtons(next.activeTool);
            this.refreshSegmentationActionButtons();
        }
        if (patch.isPinned !== undefined) {
            this.updateSegmentationPanelVisibility();
        }
        if (patch.tool !== undefined || patch.paintValue !== undefined || patch.activeTool !== undefined) {
            this.updateInteractionCursor();
        }
        if (patch.showOnlyActive !== undefined || patch.visible !== undefined || patch.activeROIId !== undefined) {
            this.scheduleSliceRender();
        }
        if (patch.showOnlyActive !== undefined
            || patch.visible !== undefined
            || patch.overlayOpacity !== undefined
            || patch.activeROIId !== undefined
            || patch.color !== undefined) {
            this.mark3DPaletteDirty();
            this.schedule3DMaskSync({ immediate: !this.segmentationDragging, render: true });
        }
        this.refreshSegmentationOverlayUI();
    }

    // ================================================================
    // Volume display setup
    // ================================================================

    private ensureMaskVolumeForCurrentVolume(): void {
        const volume = this.volume;
        if (!volume) {
            this.segmentationStore.clearMask();
            this.maskSliceBuffers = { xy: null, xz: null, yz: null };
            this.maskDisplaySliceBuffers = { xy: null, xz: null, yz: null };
            this.mask3DLookupCache = null;
            this.mask3DLabelBuffer = null;
            this.invalidateActiveRoiStats();
            this.mark3DMaskDirty();
            this.schedule3DMaskSync({ immediate: true, render: true });
            return;
        }

        const recreated = this.segmentationStore.ensureMaskVolume(volume.dimensions, {
            preferredBackend: 'auto',
            classDataType: 'uint8',
        });
        if (recreated) {
            this.maskSliceBuffers = { xy: null, xz: null, yz: null };
            this.maskDisplaySliceBuffers = { xy: null, xz: null, yz: null };
            this.mask3DLookupCache = null;
            this.mask3DLabelBuffer = null;
            this.invalidateActiveRoiStats();
            this.mark3DMaskDirty();
            this.schedule3DMaskSync({ render: false });
        }
    }

    private displayVolume(): void {
        const volume = this.volume;
        if (!volume) return;

        this.ensureMaskVolumeForCurrentVolume();

        const [nx, ny, nz] = volume.dimensions;

        // Set window to full data range
        this.displayWindowMin = volume.min;
        this.displayWindowMax = volume.max;
        for (const axis of AXES) {
            this.sliceRenderers[axis]?.setWindow(volume.min, volume.max);
        }

        // Navigate to center slices
        const centerSlices = {
            xy: Math.floor(nz / 2),
            xz: Math.floor(ny / 2),
            yz: Math.floor(nx / 2),
        };
        this.uiState.update({ slices: centerSlices });

        // Set crosshair to center
        this.crosshairPos = {
            x: Math.floor(nx / 2),
            y: Math.floor(ny / 2),
            z: Math.floor(nz / 2),
        };

        // Upload volume to 3D renderer
        // Streaming volumes: upload the 4x downsampled MIP volume ("low")
        // Standard volumes: only upload full when dimensions and memory budget are safe.
        const mipVolume = volume.getMIPVolume();
        if (this.mipRenderer) {
            const uploadOrDisable = (vol: VolumeData, resolution: 'low' | 'mid' | 'full'): boolean => {
                if (!this.mipRenderer || this.volume !== volume) return false;
                if (!this.canUploadVolumeTo3D(vol)) {
                    this.mipRenderer.unloadVolume();
                    this.mipRenderer.clear();
                    this.set3DStatusText('3D disabled: volume too large for GPU memory');
                    this.mark3DMaskDirty();
                    return false;
                }
                try {
                    this.mipRenderer.uploadVolume(vol);
                    this.current3DResolution = resolution;
                    this.update3DResolutionIndicator(vol);
                    this.set3DStatusText('');
                    this.mark3DMaskDirty();
                    this.mark3DPaletteDirty();
                    this.schedule3DMaskSync({ immediate: true, render: false });
                    return true;
                } catch (error) {
                    console.error('3D upload failed:', error);
                    this.mipRenderer.unloadVolume();
                    this.mipRenderer.clear();
                    this.set3DStatusText('3D upload failed');
                    this.mark3DMaskDirty();
                    return false;
                }
            };

            const uploadedInitial = !volume.isStreaming && this.canUploadVolumeTo3D(mipVolume)
                ? uploadOrDisable(mipVolume, 'full')
                : false;

            if (!uploadedInitial) {
                this.current3DResolution = 'low';

                if (volume.isStreaming) {
                    uploadOrDisable(mipVolume, 'low');
                } else if (mipVolume.canEnhance3D()) {
                    this.set3DStatusText('0%');
                    mipVolume.createDownsampledVolume(4, (pct) => {
                        this.set3DStatusText(`${pct}%`);
                    }).then((downsampled) => {
                        if (!downsampled || this.volume !== volume) {
                            return;
                        }
                        this.volumeStore.cache3DVolume('low', downsampled);
                        const ok = uploadOrDisable(downsampled, 'low');
                        if (ok && this.mipRenderer) {
                            this.mipRenderer.render();
                        }
                    }).catch((error) => {
                        console.error('Failed generating low-res 3D volume:', error);
                        this.set3DStatusText('3D preparation failed');
                    });
                } else {
                    uploadOrDisable(mipVolume, 'low');
                }
            }
            this.mipRenderer.resetCamera();
        }

        // Compute histogram (from MIP volume, which is always in-memory)
        this.computeHistogram(mipVolume).then(() => {
            if (this.volume !== volume) return;
            this.drawHistogram();
            this.updateHandlePositions();
        });

        // Show floating controls once a volume is loaded
        const sliceControls = document.getElementById('sliceControls');
        if (sliceControls) sliceControls.classList.add('visible');
        const quality3D = document.getElementById('quality3DGroup');
        if (quality3D) quality3D.style.display = 'block';
        this.updateSegmentationPanelVisibility();
        this.syncModeButtons();
        this.updateInteractionCursor();

        // Sync zoom/pan/crosshair to renderers
        this.syncZoomPan();
        this.updateCrosshairs();
        this.updateSliceIndicators();
        this.updatePixelInfo();

        // Update 3D resolution dropdown options
        this.update3DResolutionOptions();
        this.update3DStatusChip();
        this.refreshFooterInfoDetails();
        this.mark3DPaletteDirty();
        this.schedule3DMaskSync({ render: false });

        this.renderAll();
    }

    private estimate3DUploadBytesForDims(nx: number, ny: number, nz: number): number {
        const voxels = nx * ny * nz;
        const texelBytes = 2; // MIP volume is uploaded as r16float
        const volumeBytes = voxels * texelBytes;
        const brickCount = Math.ceil(nx / BRICK_SIZE) * Math.ceil(ny / BRICK_SIZE) * Math.ceil(nz / BRICK_SIZE);
        const brickBytes = brickCount * 4; // rg16float
        // Small safety factor for driver-side allocation overhead.
        return Math.ceil((volumeBytes + brickBytes) * 1.15);
    }

    private canUploadDimsTo3D(nx: number, ny: number, nz: number): boolean {
        const maxDim3D = this.gpu?.device.limits.maxTextureDimension3D ?? 0;
        if (nx > maxDim3D || ny > maxDim3D || nz > maxDim3D) return false;
        return this.estimate3DUploadBytesForDims(nx, ny, nz) <= MAX_SAFE_3D_UPLOAD_BYTES;
    }

    private canUploadVolumeTo3D(vol: VolumeData): boolean {
        const [nx, ny, nz] = vol.dimensions;
        return this.canUploadDimsTo3D(nx, ny, nz);
    }

    private set3DStatusText(text: string): void {
        const status = document.getElementById('resolution3DStatus');
        if (status) status.textContent = text;
    }

    /**
     * Update which 3D resolution options are enabled based on volume type and GPU limits.
     * Also updates option labels to show actual dimensions.
     */
    private update3DResolutionOptions(): void {
        const select = document.getElementById('resolution3DSelect') as HTMLSelectElement | null;
        if (!select) return;

        const volume = this.volume;
        if (!volume) return;

        const lowOption = select.querySelector('option[value="low"]') as HTMLOptionElement | null;
        const midOption = select.querySelector('option[value="mid"]') as HTMLOptionElement | null;
        const fullOption = select.querySelector('option[value="full"]') as HTMLOptionElement | null;
        if (!lowOption || !midOption || !fullOption) return;

        const [nx, ny, nz] = volume.dimensions;
        const maxDim3D = this.gpu?.device.limits.maxTextureDimension3D ?? 0;

        // Low (4x downsample)
        const lowNx = Math.ceil(nx / 4);
        const lowNy = Math.ceil(ny / 4);
        const lowNz = Math.ceil(nz / 4);
        lowOption.textContent = `Low (${lowNx}x${lowNy}x${lowNz})`;
        const canLow = this.canUploadDimsTo3D(lowNx, lowNy, lowNz);
        lowOption.disabled = !canLow;

        // Mid (2x downsample)
        const midNx = Math.ceil(nx / 2);
        const midNy = Math.ceil(ny / 2);
        const midNz = Math.ceil(nz / 2);
        const canMid = volume.canEnhance3D() &&
            midNx <= maxDim3D && midNy <= maxDim3D && midNz <= maxDim3D &&
            this.canUploadDimsTo3D(midNx, midNy, midNz);
        midOption.textContent = `Mid (${midNx}x${midNy}x${midNz})`;
        midOption.disabled = !canMid;

        // Full
        const canFull = !volume.isStreaming &&
            nx <= maxDim3D && ny <= maxDim3D && nz <= maxDim3D &&
            this.canUploadDimsTo3D(nx, ny, nz);
        fullOption.textContent = `Full (${nx}x${ny}x${nz})`;
        fullOption.disabled = !canFull;

        // If current selection is now disabled, fall back
        if ((this.current3DResolution === 'mid' && midOption.disabled) ||
            (this.current3DResolution === 'full' && fullOption.disabled)) {
            this.current3DResolution = 'low';
        }
        if (lowOption.disabled && !midOption.disabled) {
            this.current3DResolution = 'mid';
        } else if (lowOption.disabled && midOption.disabled && !fullOption.disabled) {
            this.current3DResolution = 'full';
        }
        select.value = this.current3DResolution;
        if (lowOption.disabled && midOption.disabled && fullOption.disabled) {
            this.set3DStatusText('3D disabled: memory budget exceeded');
        } else if (document.getElementById('resolution3DStatus')?.textContent === '3D disabled: memory budget exceeded') {
            this.set3DStatusText('');
        }
        this.update3DStatusChip();
    }

    /**
     * Refresh 3D resolution/status controls after a resolution change.
     */
    private update3DResolutionIndicator(_vol: VolumeData): void {
        this.update3DStatusChip();
    }

    /**
     * Switch 3D resolution mode (low/mid/full).
     * Preserves the user's current window/level settings.
     */
    private async set3DResolution(value: 'low' | 'mid' | 'full'): Promise<void> {
        if (!this.volume || !this.mipRenderer) return;

        // Token used to detect if the volume changed while awaiting async work.
        const volumeToken = this.volume;
        this.resolution3DVolumeToken = volumeToken;

        const select = document.getElementById('resolution3DSelect') as HTMLSelectElement | null;
        const status = document.getElementById('resolution3DStatus');

        // Save current window/level so uploadVolume doesn't clobber it
        const savedMin = this.mipRenderer.displayMin;
        const savedMax = this.mipRenderer.displayMax;

        /** Returns false if the volume changed during an async gap — the caller should abort. */
        const isStale = (): boolean => {
            return this.resolution3DVolumeToken !== volumeToken || this.volume !== volumeToken;
        };

        const uploadAndRender = (vol: VolumeData): boolean => {
            if (isStale()) return false;
            if (!this.canUploadVolumeTo3D(vol)) {
                this.set3DStatusText('Volume too large for GPU memory');
                return false;
            }
            try {
                this.mipRenderer!.uploadVolume(vol);
            } catch (error) {
                console.error('3D upload failed while changing resolution:', error);
                this.mipRenderer!.unloadVolume();
                this.mipRenderer!.clear();
                this.set3DStatusText('3D upload failed');
                this.mark3DMaskDirty();
                return false;
            }
            this.mark3DMaskDirty();
            this.mark3DPaletteDirty();
            this.sync3DMaskOverlayNow(false);
            // Restore user's window/level
            this.mipRenderer!.displayMin = savedMin;
            this.mipRenderer!.displayMax = savedMax;
            this.mipRenderer!.render();
            // Update 3D resolution indicator
            this.update3DResolutionIndicator(vol);
            return true;
        };

        const clearStatus = () => {
            if (status) status.textContent = '';
        };

        const revertSelect = () => {
            if (select) {
                select.disabled = false;
                select.value = this.current3DResolution;
            }
            clearStatus();
            this.update3DStatusChip();
        };

        try {
            if (value === 'low') {
                if (this.volume.isStreaming) {
                    // Streaming: getMIPVolume() is the 4x downsampled lowRes
                    if (!uploadAndRender(this.volume.getMIPVolume())) {
                        revertSelect();
                        return;
                    }
                } else {
                    // Standard: generate 4x downsample on demand
                    const cached = this.volumeStore.getCached3DVolume('low');
                    if (cached) {
                        if (!uploadAndRender(cached)) {
                            revertSelect();
                            return;
                        }
                    } else {
                        if (select) select.disabled = true;
                        if (status) status.textContent = '0%';
                        const lowVol = await (this.volume as VolumeData).createDownsampledVolume(4, (pct) => {
                            if (!isStale() && status) status.textContent = `${pct}%`;
                        });
                        if (isStale()) { revertSelect(); return; }
                        if (select) select.disabled = false;
                        clearStatus();
                        if (lowVol) {
                            this.volumeStore.cache3DVolume('low', lowVol);
                            if (!uploadAndRender(lowVol)) {
                                revertSelect();
                                return;
                            }
                        } else {
                            revertSelect();
                            return;
                        }
                    }
                }
                this.current3DResolution = 'low';

            } else if (value === 'mid') {
                const cached = this.volumeStore.getCached3DVolume('mid');
                if (cached) {
                    if (!uploadAndRender(cached)) {
                        revertSelect();
                        return;
                    }
                } else {
                    if (!this.volume.canEnhance3D()) {
                        revertSelect();
                        return;
                    }
                    if (select) select.disabled = true;
                    if (status) status.textContent = '0%';
                    const midVol = await this.volume.createDownsampledVolume(2, (pct) => {
                        if (!isStale() && status) status.textContent = `${pct}%`;
                    });
                    if (isStale()) { revertSelect(); return; }
                    if (select) select.disabled = false;
                    clearStatus();
                    if (midVol) {
                        this.volumeStore.cache3DVolume('mid', midVol);
                        if (!uploadAndRender(midVol)) {
                            revertSelect();
                            return;
                        }
                    } else {
                        revertSelect();
                        return;
                    }
                }
                this.current3DResolution = 'mid';

            } else if (value === 'full') {
                if (this.volume.isStreaming) {
                    revertSelect();
                    return;
                }
                if (!uploadAndRender(this.volume as VolumeData)) {
                    revertSelect();
                    return;
                }
                this.current3DResolution = 'full';
            }
            this.update3DStatusChip();
        } catch (error) {
            console.error('Failed to set 3D resolution:', error);
            revertSelect();
        }
    }

    private updateSliceIndicators(): void {
        if (!this.volume) return;
        const [nx, ny, nz] = this.volume.dimensions;
        const s = this.uiState.state.slices;

        this.sliceIndicators.xy.textContent = `XY: ${s.xy + 1}/${nz}`;
        this.sliceIndicators.xz.textContent = `XZ: ${s.xz + 1}/${ny}`;
        this.sliceIndicators.yz.textContent = `YZ: ${s.yz + 1}/${nx}`;
        this.syncThresholdRangeUI(this.uiState.state.segmentation);
    }

    private maxSlice(axis: ViewAxis): number {
        if (!this.volume) return 0;
        const [nx, ny, nz] = this.volume.dimensions;
        switch (axis) {
            case 'xy': return nz - 1;
            case 'xz': return ny - 1;
            case 'yz': return nx - 1;
        }
    }

    // ================================================================
    // File handling
    // ================================================================

    private async handleFiles(groups: FileGroup[]): Promise<void> {
        if (groups.length === 0) return;

        const group = groups[0];
        const name = group.name;

        // Reset 3D resolution cache for new volume
        this.clear3DMaskSyncTimer();
        this.mask3DSyncWantsRender = false;
        this.resolution3DVolumeToken = null; // Invalidate any in-flight resolution change
        this.volumeStore.clear({ disposeStreaming: true, resetResolution: true });
        this.segmentationStore.clearMask();
        this.maskSliceBuffers = { xy: null, xz: null, yz: null };
        this.maskDisplaySliceBuffers = { xy: null, xz: null, yz: null };
        this.mask3DLookupCache = null;
        this.mask3DLabelBuffer = null;
        this.mark3DMaskDirty();
        this.mark3DPaletteDirty();
        this.segmentationDragging = false;
        this.segmentationDragAxis = null;
        this.activePaintStrokeId = null;
        this.segmentationWorkerBusy = false;
        this.thresholdGuideCacheKey = null;
        this.thresholdGuideCacheValue = null;
        this.smartRegionService?.clearEmbeddingCache();
        this.clearSmartRegionPreview();
        this.setSmartRegionStatus(SMART_REGION_STATUS_DEFAULT);
        this.resetRoiEntries();

        const seg = this.uiState.state.segmentation;
        this.uiState.update({
            appMode: AppMode.Viewing,
            segmentation: {
                ...seg,
                enabled: false,
                visible: true,
                overlayOpacity: 0.4,
                color: [1.0, 0.0, 0.0],
                activeROIId: null,
                activeTool: 'brush',
                showOnlyActive: false,
                aiPreviewMask: null,
                isPinned: seg.isPinned,
                activeClassId: 1,
                tool: 'brush',
                brushRadius: 8,
                paintValue: 1,
                thresholdMin: 0,
                thresholdMax: 0,
                regionGrowTolerance: 25,
                regionGrowSliceRadius: 1,
            },
        });
        this.updateSegmentationToolRows('brush');
        this.updateSegmentationToolButtons('brush');
        this.syncModeButtons();
        this.updateSegmentationPanelVisibility();
        this.updateInteractionCursor();
        this.schedule3DMaskSync({ immediate: true, render: true });

        this.uiState.setLoading(true);
        this.showLoadingOverlay('Loading...');

        try {
            const volume = await loadVolume(
                group,
                (stage, pct) => {
                    const pctText = pct !== undefined ? ` ${pct}%` : '';
                    this.updateLoadingOverlay(`${stage}${pctText}`);
                },
                // onVolumeSwap: hybrid mode background load completed
                (fullVolume) => {
                    console.log('Full volume loaded, swapping from streaming');
                    this.volumeStore.replaceVolume(fullVolume, {
                        disposePreviousStreaming: true,
                        reset3DCache: true,
                        resetResolution: true,
                    });
                    const fullInfo = fullVolume.getInfo();
                    this.fileNameEl.textContent = name;
                    const [w, h, d] = fullInfo.dimensions;
                    this.imageInfoEl.textContent =
                        `${w}x${h}x${d} ${fullInfo.dataType} ${fullInfo.memorySizeMB}MB`;
                    this.refreshFooterInfoDetails();
                    const shouldAutoTuneTolerance = !this.uiState.state.segmentation.enabled;
                    this.setSegmentationState({
                        thresholdMin: fullVolume.min,
                        thresholdMax: fullVolume.max,
                        ...(shouldAutoTuneTolerance
                            ? { regionGrowTolerance: this.computeAutoRegionGrowTolerance(fullVolume.min, fullVolume.max) }
                            : {}),
                    });
                    this.displayVolume();
                    this.hideLoadingOverlay();
                },
            );

            this.volumeStore.replaceVolume(volume, {
                disposePreviousStreaming: false,
                reset3DCache: false,
                resetResolution: false,
            });

            // Wire streaming callbacks if applicable
            if (volume.isStreaming) {
                this.wireStreamingCallbacks(volume as StreamingVolumeData);
            }

            const info = volume.getInfo();
            console.log('Volume loaded:', info);

            const suffix = volume.isStreaming ? ' (Streaming)' : '';
            this.fileNameEl.textContent = name + suffix;
            const [w, h, d] = info.dimensions;
            this.imageInfoEl.textContent =
                `${w}x${h}x${d} ${info.dataType} ${info.memorySizeMB}MB${suffix}`;
            this.refreshFooterInfoDetails();

            this.setSegmentationState({
                thresholdMin: volume.min,
                thresholdMax: volume.max,
                regionGrowTolerance: this.computeAutoRegionGrowTolerance(volume.min, volume.max),
            });
            this.displayVolume();
            this.uiState.setFileLoaded(name);
        } catch (err) {
            console.error('Failed to load volume:', err);
            const msg = err instanceof Error ? err.message : String(err);
            this.imageInfoEl.textContent = `Error: ${msg}`;
            this.fileNameEl.textContent = name;
            this.refreshFooterInfoDetails();
            this.showErrorOverlay(msg);
            this.uiState.setLoading(false);
            return;
        }
        this.uiState.setLoading(false);
        this.hideLoadingOverlay();
    }

    /** Connect StreamingVolumeData's onSliceReady callback to re-render the affected axis */
    private wireStreamingCallbacks(streaming: StreamingVolumeData): void {
        streaming.onSliceReady = (axis, index) => {
            // Only re-render if this slice is still the currently displayed one
            if (this.volume !== streaming) return;
            const currentIndex = this.uiState.state.slices[axis];
            if (currentIndex === index) {
                this.renderSlice(axis);
            }
        };
    }

    private showLoadingOverlay(text: string): void {
        this.hideLoadingOverlay();
        const overlay = document.createElement('div');
        overlay.className = 'loading-overlay';
        overlay.id = 'loadingOverlay';
        overlay.innerHTML = `
            <div class="loading-spinner"></div>
            <div class="loading-text" id="loadingText">${text}</div>
        `;
        this.imageWrapper.appendChild(overlay);
    }

    private updateLoadingOverlay(text: string): void {
        const el = document.getElementById('loadingText');
        if (el) el.textContent = text;
    }

    private hideLoadingOverlay(): void {
        document.getElementById('loadingOverlay')?.remove();
    }

    private showErrorOverlay(message: string): void {
        this.hideLoadingOverlay();
        const overlay = document.createElement('div');
        overlay.className = 'loading-overlay';
        overlay.id = 'loadingOverlay';
        overlay.innerHTML = `
            <div style="color: #ff6b6b; font-size: 2rem; margin-bottom: 1rem;">&#x26A0;</div>
            <div class="loading-text" style="color: #ff6b6b;">Failed to load file</div>
            <div style="color: #999; font-size: 0.85rem; margin-top: 0.5rem; max-width: 400px; text-align: center;">${message}</div>
            <button style="margin-top: 1rem; padding: 0.5rem 1rem; cursor: pointer;" onclick="this.parentElement.remove()">Dismiss</button>
        `;
        this.imageWrapper.appendChild(overlay);
    }

    // ================================================================
    // 2D viewport interactions (pan, zoom, scroll, crosshair, maximize)
    // ================================================================

    private shouldUseBrushTool(modifiers?: { shiftKey?: boolean; ctrlKey?: boolean; metaKey?: boolean }): boolean {
        const seg = this.uiState.state.segmentation;
        const active = this.getActiveRoi();
        const paintValue = this.getEffectiveBrushPaintValue(modifiers);
        return !!this.maskVolume
            && this.isSegmentationMode()
            && seg.enabled
            && seg.tool === 'brush'
            && (paintValue === 0 || (!!active && !active.locked));
    }

    private getEffectiveBrushPaintValue(modifiers?: { shiftKey?: boolean; ctrlKey?: boolean; metaKey?: boolean }): 0 | 1 {
        if (modifiers?.shiftKey) return 0;
        if (modifiers?.ctrlKey || modifiers?.metaKey) return 1;
        return this.uiState.state.segmentation.paintValue;
    }

    private getActionShortcutOverrideFromModifiers(modifiers?: { shiftKey?: boolean; ctrlKey?: boolean; metaKey?: boolean }): 0 | 1 | null {
        if (modifiers?.shiftKey) return 0;
        if (modifiers?.ctrlKey || modifiers?.metaKey) return 1;
        return null;
    }

    private updateActionShortcutOverride(modifiers?: { shiftKey?: boolean; ctrlKey?: boolean; metaKey?: boolean }): void {
        const next = this.getActionShortcutOverrideFromModifiers(modifiers);
        if (next === this.actionShortcutOverride) return;
        this.actionShortcutOverride = next;
        this.refreshSegmentationActionButtons();
    }

    private shouldUseThresholdTool(modifiers?: { shiftKey?: boolean; ctrlKey?: boolean; metaKey?: boolean }): boolean {
        const seg = this.uiState.state.segmentation;
        const active = this.getActiveRoi();
        const paintValue = this.getEffectiveBrushPaintValue(modifiers);
        const canErase = paintValue === 0;
        const canPaint = paintValue === 1 && !!active && !active.locked;
        return !!this.maskVolume
            && this.isSegmentationMode()
            && seg.enabled
            && seg.tool === 'threshold'
            && (canErase || canPaint);
    }

    private shouldUseRegionGrowTool(modifiers?: { shiftKey?: boolean; ctrlKey?: boolean; metaKey?: boolean }): boolean {
        const seg = this.uiState.state.segmentation;
        const active = this.getActiveRoi();
        const paintValue = this.getEffectiveBrushPaintValue(modifiers);
        const canErase = paintValue === 0;
        const canPaint = paintValue === 1 && !!active && !active.locked;
        return !!this.maskVolume
            && this.isSegmentationMode()
            && seg.enabled
            && seg.tool === 'region-grow'
            && (canErase || canPaint);
    }

    private shouldUseSmartRegionTool(): boolean {
        const seg = this.uiState.state.segmentation;
        const active = this.getActiveRoi();
        return !!this.maskVolume
            && this.isSegmentationMode()
            && seg.enabled
            && seg.tool === 'smart-region'
            && !!active
            && !active.locked;
    }

    private getSmartRegionService(): Sam2SliceService {
        if (!this.smartRegionService) {
            this.smartRegionService = new Sam2SliceService();
        }
        this.updateSmartRegionBackendChip(this.smartRegionService.isUsingWasmFallback());
        return this.smartRegionService;
    }

    private buildSmartRegionVolumeKey(volume: VolumeData | StreamingVolumeData): string {
        return `${this.uiState.state.fileName}|${volume.dimensions.join('x')}|${volume.dataType}`;
    }

    private getSmartRegionWindow(volume: VolumeData | StreamingVolumeData): { windowMin: number; windowMax: number } {
        const windowMinCandidate = Math.min(this.displayWindowMin, this.displayWindowMax);
        const windowMaxCandidate = Math.max(this.displayWindowMin, this.displayWindowMax);
        const windowMin = Number.isFinite(windowMinCandidate) ? windowMinCandidate : volume.min;
        const windowMax = Number.isFinite(windowMaxCandidate) && windowMaxCandidate > windowMin
            ? windowMaxCandidate
            : Math.max(windowMin + 1e-6, volume.max);
        return { windowMin, windowMax };
    }

    private setSmartRegionStatus(message: string, isError = false): void {
        if (!this.segAIStatusEl) return;
        this.segAIStatusEl.textContent = message;
        this.segAIStatusEl.classList.toggle('error', isError);
    }

    private updateSmartRegionBackendChip(isWasmFallback: boolean): void {
        if (!this.segAIBackendChipEl) return;
        this.segAIBackendChipEl.style.display = isWasmFallback ? '' : 'none';
        this.segAIBackendChipEl.textContent = isWasmFallback ? 'CPU only' : '';
        this.segAIBackendChipEl.title = isWasmFallback
            ? 'SAM2 is running in WASM/CPU fallback mode because WebGPU failed on this system.'
            : '';
    }

    private refreshSmartRegionPreviewControls(): void {
        if (this.segAIPreviewToggle) {
            this.segAIPreviewToggle.checked = this.smartRegionPreviewOnly;
        }
        if (this.segAIApplyPreviewBtn) {
            this.segAIApplyPreviewBtn.disabled = !this.smartRegionPreview || this.segmentationWorkerBusy;
        }
        if (this.segAIClearPreviewBtn) {
            this.segAIClearPreviewBtn.disabled = !this.smartRegionPreview || this.segmentationWorkerBusy;
        }
    }

    private getSmartRegionPromptLayer(axis: ViewAxis): HTMLDivElement | null {
        const existing = this.smartRegionPromptLayers[axis];
        if (existing) return existing;
        const container = this.viewportContainers[axis];
        if (!container) return null;
        const layer = document.createElement('div');
        layer.className = 'smart-prompt-layer';
        container.appendChild(layer);
        this.smartRegionPromptLayers[axis] = layer;
        return layer;
    }

    private clearSmartRegionPromptMarkers(axis?: ViewAxis): void {
        const axes = axis ? [axis] : AXES;
        for (const a of axes) {
            const layer = this.smartRegionPromptLayers[a];
            if (!layer) continue;
            layer.textContent = '';
            layer.style.display = 'none';
        }
    }

    private updateSmartRegionPromptMarkers(axis: ViewAxis, sliceIndex: number, renderer: SliceRenderer): void {
        const layer = this.getSmartRegionPromptLayer(axis);
        if (!layer) return;
        const preview = this.smartRegionPreview;
        if (!preview || preview.axis !== axis || preview.sliceIndex !== sliceIndex || preview.points.length === 0) {
            layer.textContent = '';
            layer.style.display = 'none';
            return;
        }

        const canvas = this.sliceCanvases[axis];
        if (!canvas || canvas.clientWidth <= 0 || canvas.clientHeight <= 0) {
            layer.textContent = '';
            layer.style.display = 'none';
            return;
        }

        const dpr = canvas.width / canvas.clientWidth;
        if (!Number.isFinite(dpr) || dpr <= 0) {
            layer.textContent = '';
            layer.style.display = 'none';
            return;
        }

        layer.textContent = '';
        layer.style.display = '';
        for (let i = 0; i < preview.points.length; i++) {
            const point = preview.points[i];
            const [canvasX, canvasY] = renderer.sliceToCanvas(point.x + 0.5, point.y + 0.5);
            const cssX = canvasX / dpr;
            const cssY = canvasY / dpr;
            if (!Number.isFinite(cssX) || !Number.isFinite(cssY)) continue;

            const marker = document.createElement('div');
            marker.className = `smart-prompt-marker ${point.label === 1 ? 'positive' : 'negative'}`;
            marker.textContent = point.label === 1 ? '+' : '-';
            marker.style.left = `${cssX}px`;
            marker.style.top = `${cssY}px`;
            layer.appendChild(marker);
        }
    }

    private clearSmartRegionPreview(options?: { render?: boolean; resetStatus?: boolean }): void {
        const prev = this.smartRegionPreview;
        if (!prev && !options?.resetStatus) {
            this.clearSmartRegionPromptMarkers();
            return;
        }
        this.smartRegionPreview = null;
        this.maskPreviewSliceBuffers = { xy: null, xz: null, yz: null };
        this.clearSmartRegionPromptMarkers();
        this.refreshSmartRegionPreviewControls();
        if (options?.resetStatus) {
            this.setSmartRegionStatus(SMART_REGION_STATUS_DEFAULT);
        }
        if (options?.render && prev) {
            this.scheduleAxisRender(prev.axis);
        }
    }

    private setSmartRegionPreview(preview: SmartRegionPreview): void {
        this.smartRegionPreview = preview;
        this.maskPreviewSliceBuffers = { xy: null, xz: null, yz: null };
        this.refreshSmartRegionPreviewControls();
        this.scheduleAxisRender(preview.axis);
    }

    private async applySmartRegionPreview(): Promise<number> {
        const preview = this.smartRegionPreview;
        if (!preview) return 0;
        let centerSelectedIndices = preview.selectedIndices;
        let refined = false;
        const volume = this.volume;
        const active = this.getActiveRoi();
        const configuredRadius = this.uiState.state.segmentation.regionGrowSliceRadius;
        const targetSlices = volume
            ? this.getSliceIndicesWithRadius(preview.axis, preview.sliceIndex, configuredRadius)
            : [preview.sliceIndex];
        const canRefine =
            preview.qualityUsed === 'preview'
            && !!volume
            && !!active
            && !active.locked
            && preview.volumeKey === this.buildSmartRegionVolumeKey(volume);
        const needsSliceRadiusInference = targetSlices.length > 1;
        const shouldRunApplyInference = canRefine || needsSliceRadiusInference;
        let changed = 0;

        if (shouldRunApplyInference && volume && active) {
            const lastPoint = preview.points[preview.points.length - 1] ?? {
                x: Math.floor(preview.width / 2),
                y: Math.floor(preview.height / 2),
                label: 1 as 0 | 1,
            };
            const applyQuality: Sam2InferenceQuality = canRefine ? 'full' : preview.qualityUsed;
            this.segmentationWorkerBusy = true;
            this.setActiveRoiBusy(active.id, true);
            this.refreshSmartRegionPreviewControls();
            this.setSmartRegionStatus(
                canRefine
                    ? 'SAM2 refining full quality...'
                    : `SAM2 applying across ${targetSlices.length} slices...`,
            );
            try {
                const service = this.getSmartRegionService();
                this.updateSmartRegionBackendChip(service.isUsingWasmFallback());
                for (const sliceIndex of targetSlices) {
                    const slice = volume.getSlice(preview.axis, sliceIndex);
                    const values = slice.data instanceof Float32Array ? slice.data : new Float32Array(slice.data);
                    const result = await service.segmentFromClick({
                        volumeKey: preview.volumeKey,
                        axis: preview.axis,
                        sliceIndex,
                        width: slice.width,
                        height: slice.height,
                        values,
                        pointX: lastPoint.x,
                        pointY: lastPoint.y,
                        pointLabel: lastPoint.label,
                        points: preview.points,
                        windowMin: preview.windowMin,
                        windowMax: preview.windowMax,
                        inferenceQuality: applyQuality,
                    });
                    if (sliceIndex === preview.sliceIndex) {
                        centerSelectedIndices = result.selectedIndices;
                    }
                    changed += this.applySelectionIndicesToActiveRoi(
                        preview.axis,
                        sliceIndex,
                        slice.width,
                        slice.height,
                        result.selectedIndices,
                    );
                }
                refined = canRefine;
            } catch (error) {
                console.warn('SAM2 apply-time inference failed; using available preview result.', error);
            } finally {
                this.segmentationWorkerBusy = false;
                this.setActiveRoiBusy(active.id, false);
                this.refreshSmartRegionPreviewControls();
            }
        }

        if (changed === 0) {
            changed = this.applySelectionIndicesToActiveRoi(
                preview.axis,
                preview.sliceIndex,
                preview.width,
                preview.height,
                centerSelectedIndices,
            );
        }
        this.clearSmartRegionPreview({ render: true });
        if (changed > 0) {
            this.invalidateActiveRoiStats();
            this.scheduleActiveRoiStatsRefresh({ force: true });
            this.updatePixelInfo();
            this.schedule3DMaskSync({ immediate: true, render: true });
        }
        this.setSmartRegionStatus(
            `SAM2 ${refined ? 'refined+applied' : 'preview applied'}: ${centerSelectedIndices.length.toLocaleString()} voxels (${changed.toLocaleString()} changed, ${targetSlices.length} slices).`,
        );
        return changed;
    }

    private applySmartRegionPreviewToMaskSlice(
        axis: ViewAxis,
        sliceIndex: number,
        maskSlice: MaskTypedArray | null,
    ): MaskTypedArray | null {
        const preview = this.smartRegionPreview;
        if (!preview || preview.axis !== axis || preview.sliceIndex !== sliceIndex || !maskSlice) {
            return maskSlice;
        }

        const ctor = maskSlice instanceof Uint16Array ? Uint16Array : Uint8Array;
        let out = this.maskPreviewSliceBuffers[axis];
        if (!out || out.length !== maskSlice.length || out.constructor !== ctor) {
            out = new ctor(maskSlice.length) as MaskTypedArray;
            this.maskPreviewSliceBuffers[axis] = out;
        }
        out.set(maskSlice);
        const classId = Math.max(1, Math.min(255, preview.classId));
        const indices = preview.selectedIndices;
        for (let i = 0; i < indices.length; i++) {
            const idx = indices[i];
            if (idx < out.length) {
                out[idx] = classId;
            }
        }
        return out;
    }

    private getSliceHitFromClient(axis: ViewAxis, clientX: number, clientY: number): SliceHit | null {
        const renderer = this.sliceRenderers[axis];
        const canvas = this.sliceCanvases[axis];
        if (!renderer || !canvas) return null;
        const rect = canvas.getBoundingClientRect();
        const dpr = canvas.width / canvas.clientWidth;
        const canvasX = (clientX - rect.left) * dpr;
        const canvasY = (clientY - rect.top) * dpr;
        const [sliceX, sliceY] = renderer.canvasToSlice(canvasX, canvasY);
        if (sliceX < 0 || sliceY < 0) return null;
        return {
            renderer,
            sliceIndex: this.uiState.state.slices[axis],
            sliceX,
            sliceY,
        };
    }

    private applySegmentationOp(op: SegmentationOp): number {
        const changed = this.segmentationStore.applyOp(op);
        if (changed > 0) {
            this.mark3DMaskDirty();
        }
        return changed;
    }

    private setActiveRoiBusy(roiId: string | null, busy: boolean): void {
        const roi = this.findRoiById(roiId);
        if (!roi || roi.aiBusy === busy) return;
        roi.aiBusy = busy;
        this.refreshSegmentationOverlayUI();
    }

    private applySelectionIndicesToClass(
        axis: ViewAxis,
        sliceIndex: number,
        width: number,
        height: number,
        selectedIndices: Uint32Array,
        classId: number,
    ): number {
        const op = createApplySliceSelectionOp({
            axis,
            sliceIndex,
            width,
            height,
            selectedIndices,
            classId,
        });
        return this.applySegmentationOp(op);
    }

    private applySelectionIndicesToActiveRoi(
        axis: ViewAxis,
        sliceIndex: number,
        width: number,
        height: number,
        selectedIndices: Uint32Array,
    ): number {
        const active = this.getActiveRoi();
        if (!active) return 0;
        return this.applySelectionIndicesToClass(axis, sliceIndex, width, height, selectedIndices, active.classId);
    }

    private getSliceIndicesWithRadius(axis: ViewAxis, centerSlice: number, radius: number): number[] {
        const clampedRadius = Math.max(0, Math.min(MAX_SEGMENTATION_SLICE_RADIUS, Math.floor(radius)));
        const minSlice = Math.max(0, centerSlice - clampedRadius);
        const maxSlice = Math.min(this.maxSlice(axis), centerSlice + clampedRadius);
        const slices: number[] = [];
        for (let slice = minSlice; slice <= maxSlice; slice++) {
            slices.push(slice);
        }
        return slices;
    }

    private paintBrushAtClient(
        axis: ViewAxis,
        clientX: number,
        clientY: number,
        modifiers?: { shiftKey?: boolean; ctrlKey?: boolean; metaKey?: boolean },
    ): number {
        const hit = this.getSliceHitFromClient(axis, clientX, clientY);
        if (!hit) return 0;
        const seg = this.uiState.state.segmentation;
        const active = this.getActiveRoi();
        const dragging = this.segmentationDragging;
        const paintValue = this.getEffectiveBrushPaintValue(modifiers);
        if (paintValue !== 0 && (!active || active.locked)) return 0;
        const classId = paintValue === 0 ? 0 : active!.classId;
        const mergeKey = this.activePaintStrokeId != null ? `paint-stroke:${this.activePaintStrokeId}` : undefined;
        const op = createPaintStrokeOp({
            axis,
            sliceIndex: hit.sliceIndex,
            sliceX: hit.sliceX,
            sliceY: hit.sliceY,
            radius: seg.brushRadius,
            classId,
            sliceRadius: seg.regionGrowSliceRadius,
            mergeKey,
            onVoxelChanged: dragging
                ? undefined
                : (x, y, z, previousClassId, nextClassId) => {
                    this.applyVoxelChangeToActiveRoiStats(x, y, z, previousClassId, nextClassId);
                },
        });
        const changed = this.applySegmentationOp(op);
        if (changed > 0) {
            this.scheduleAxisRender(axis);
            if (dragging) {
                this.activeRoiStatsDirty = true;
            } else {
                this.updatePixelInfo();
                this.scheduleActiveRoiStatsRefresh();
                this.schedule3DMaskSync({ render: true });
            }
        }
        return changed;
    }

    private async runThresholdToolAtClient(
        axis: ViewAxis,
        clientX: number,
        clientY: number,
        modifiers?: { shiftKey?: boolean; ctrlKey?: boolean; metaKey?: boolean },
    ): Promise<void> {
        if (this.segmentationWorkerBusy || !this.volume) return;
        const hit = this.getSliceHitFromClient(axis, clientX, clientY);
        if (!hit) return;
        const active = this.getActiveRoi();
        const paintValue = this.getEffectiveBrushPaintValue(modifiers);
        const eraseMode = paintValue === 0;
        if (!eraseMode && (!active || active.locked)) return;
        const classId = eraseMode ? 0 : active!.classId;
        const seg = this.uiState.state.segmentation;
        this.segmentationWorkerBusy = true;
        if (active && !eraseMode) {
            this.setActiveRoiBusy(active.id, true);
        }
        try {
            const targetSlices = this.getSliceIndicesWithRadius(axis, hit.sliceIndex, seg.regionGrowSliceRadius);
            let changedTotal = 0;
            for (const sliceIndex of targetSlices) {
                const slice = this.volume.getSlice(axis, sliceIndex);
                const values = slice.data instanceof Float32Array ? slice.data : new Float32Array(slice.data);
                const selected = await this.segmentationWorker.runThresholdSlice({
                    width: slice.width,
                    height: slice.height,
                    values,
                    min: Math.min(seg.thresholdMin, seg.thresholdMax),
                    max: Math.max(seg.thresholdMin, seg.thresholdMax),
                });
                changedTotal += this.applySelectionIndicesToClass(axis, sliceIndex, slice.width, slice.height, selected, classId);
            }
            if (changedTotal > 0) {
                this.invalidateActiveRoiStats();
                this.scheduleAxisRender(axis);
                this.scheduleActiveRoiStatsRefresh({ force: true });
                this.updatePixelInfo();
                this.schedule3DMaskSync({ immediate: true, render: true });
            }
        } catch (error) {
            console.error('Threshold tool failed:', error);
        } finally {
            this.segmentationWorkerBusy = false;
            if (active && !eraseMode) {
                this.setActiveRoiBusy(active.id, false);
            }
            this.refreshSmartRegionPreviewControls();
        }
    }

    private async runRegionGrowToolAtClient(
        axis: ViewAxis,
        clientX: number,
        clientY: number,
        modifiers?: { shiftKey?: boolean; ctrlKey?: boolean; metaKey?: boolean },
    ): Promise<void> {
        if (this.segmentationWorkerBusy || !this.volume) return;
        const hit = this.getSliceHitFromClient(axis, clientX, clientY);
        if (!hit) return;
        const active = this.getActiveRoi();
        const paintValue = this.getEffectiveBrushPaintValue(modifiers);
        const eraseMode = paintValue === 0;
        if (!eraseMode && (!active || active.locked)) return;
        const classId = eraseMode ? 0 : active!.classId;
        const seg = this.uiState.state.segmentation;
        this.segmentationWorkerBusy = true;
        if (active && !eraseMode) {
            this.setActiveRoiBusy(active.id, true);
        }
        const callStartedAt = performance.now();
        const callStats: Record<RegionGrowBackend, RegionGrowPerfStats> = {
            webgpu: createRegionGrowPerfStats(),
            worker: createRegionGrowPerfStats(),
        };
        let changedTotal = 0;
        let targetSliceCount = 0;
        try {
            const targetSlices = this.getSliceIndicesWithRadius(axis, hit.sliceIndex, seg.regionGrowSliceRadius);
            targetSliceCount = targetSlices.length;
            for (const sliceIndex of targetSlices) {
                const slice = this.volume.getSlice(axis, sliceIndex);
                const values = slice.data instanceof Float32Array ? slice.data : new Float32Array(slice.data);
                const sx = Math.max(0, Math.min(slice.width - 1, Math.floor(hit.sliceX)));
                const sy = Math.max(0, Math.min(slice.height - 1, Math.floor(hit.sliceY)));
                const seedIndex = sy * slice.width + sx;
                const result = await this.runRegionGrowSlice({
                    width: slice.width,
                    height: slice.height,
                    values,
                    seedIndex,
                    tolerance: Math.max(0, seg.regionGrowTolerance),
                });
                const stats = callStats[result.backend];
                stats.slices += 1;
                stats.selected += result.selected.length;
                stats.elapsedMs += result.elapsedMs;
                changedTotal += this.applySelectionIndicesToClass(axis, sliceIndex, slice.width, slice.height, result.selected, classId);
            }
            this.logRegionGrowPerf(axis, targetSliceCount, changedTotal, performance.now() - callStartedAt, callStats);
            if (changedTotal > 0) {
                this.invalidateActiveRoiStats();
                this.scheduleAxisRender(axis);
                this.scheduleActiveRoiStatsRefresh({ force: true });
                this.updatePixelInfo();
                this.schedule3DMaskSync({ immediate: true, render: true });
            }
        } catch (error) {
            console.error('Region grow tool failed:', error);
        } finally {
            this.segmentationWorkerBusy = false;
            if (active && !eraseMode) {
                this.setActiveRoiBusy(active.id, false);
            }
            this.refreshSmartRegionPreviewControls();
        }
    }

    private async runSmartRegionToolAtClient(
        axis: ViewAxis,
        clientX: number,
        clientY: number,
        modifiers?: { shiftKey?: boolean; ctrlKey?: boolean; metaKey?: boolean },
    ): Promise<void> {
        if (this.segmentationWorkerBusy || !this.volume) return;
        const hit = this.getSliceHitFromClient(axis, clientX, clientY);
        if (!hit) return;
        const active = this.getActiveRoi();
        if (!active || active.locked) return;

        const pointLabel = this.getEffectiveBrushPaintValue(modifiers);
        const volume = this.volume;
        const slice = volume.getSlice(axis, hit.sliceIndex);
        const values = slice.data instanceof Float32Array ? slice.data : new Float32Array(slice.data);
        const pointX = Math.max(0, Math.min(slice.width - 1, Math.floor(hit.sliceX)));
        const pointY = Math.max(0, Math.min(slice.height - 1, Math.floor(hit.sliceY)));
        const clickedPoint: Sam2PromptPoint = { x: pointX, y: pointY, label: pointLabel };
        const volumeKey = this.buildSmartRegionVolumeKey(volume);
        const { windowMin, windowMax } = this.getSmartRegionWindow(volume);
        const seg = this.uiState.state.segmentation;
        const targetSlices = this.getSliceIndicesWithRadius(axis, hit.sliceIndex, seg.regionGrowSliceRadius);
        const inferenceQuality: Sam2InferenceQuality = this.smartRegionPreviewOnly ? 'preview' : 'full';
        let promptPoints: Sam2PromptPoint[] = [clickedPoint];
        if (this.smartRegionPreviewOnly) {
            const existing = this.smartRegionPreview;
            const sameContext = !!existing
                && existing.axis === axis
                && existing.sliceIndex === hit.sliceIndex
                && existing.width === slice.width
                && existing.height === slice.height
                && existing.volumeKey === volumeKey
                && Math.abs(existing.windowMin - windowMin) <= 1e-6
                && Math.abs(existing.windowMax - windowMax) <= 1e-6;
            if (sameContext) {
                promptPoints = existing.points.slice();
                promptPoints.push(clickedPoint);
            }
        }

        this.segmentationWorkerBusy = true;
        this.setActiveRoiBusy(active.id, true);
        this.refreshSmartRegionPreviewControls();
        const startedAt = performance.now();
        try {
            const service = this.getSmartRegionService();
            if (!service.isInitialized()) {
                this.setSmartRegionStatus('SAM2 loading model files (first run may take a while)...');
            } else {
                this.setSmartRegionStatus('SAM2 running...');
            }
            const result = await service.segmentFromClick({
                volumeKey,
                axis,
                sliceIndex: hit.sliceIndex,
                width: slice.width,
                height: slice.height,
                values,
                pointX,
                pointY,
                pointLabel,
                points: promptPoints,
                windowMin,
                windowMax,
                inferenceQuality,
            });
            this.updateSmartRegionBackendChip(service.isUsingWasmFallback());
            const totalMs = performance.now() - startedAt;
            const cacheTag = result.embeddingCacheHit ? 'cache' : 'encoded';
            const promptTag = pointLabel === 1 ? '+' : '-';
            const qualityTag = result.qualityUsed === 'preview' ? 'fast' : 'full';

            let changed = 0;
            if (this.smartRegionPreviewOnly) {
                this.setSmartRegionPreview({
                    axis,
                    sliceIndex: hit.sliceIndex,
                    sliceRadius: seg.regionGrowSliceRadius,
                    width: slice.width,
                    height: slice.height,
                    classId: active.classId,
                    volumeKey,
                    points: promptPoints,
                    windowMin,
                    windowMax,
                    iouScore: result.iouScore,
                    qualityUsed: result.qualityUsed,
                    selectedIndices: result.selectedIndices,
                });
                this.setSmartRegionStatus(
                    `SAM2 ${promptTag} preview (${qualityTag}, ${promptPoints.length} pts): ${result.selectedIndices.length.toLocaleString()} voxels (${cacheTag}, ${totalMs.toFixed(0)}ms).`,
                );
            } else {
                this.clearSmartRegionPreview({ render: true });
                let selectedTotal = 0;
                let iouScore = 0;
                let encodeMs = 0;
                let decodeMs = 0;
                let cacheHitCount = 0;

                for (const sliceIndex of targetSlices) {
                    let sliceResult = result;
                    let sliceData = slice;
                    if (sliceIndex !== hit.sliceIndex) {
                        sliceData = volume.getSlice(axis, sliceIndex);
                        const sliceValues = sliceData.data instanceof Float32Array ? sliceData.data : new Float32Array(sliceData.data);
                        const slicePointX = Math.max(0, Math.min(sliceData.width - 1, pointX));
                        const slicePointY = Math.max(0, Math.min(sliceData.height - 1, pointY));
                        sliceResult = await service.segmentFromClick({
                            volumeKey,
                            axis,
                            sliceIndex,
                            width: sliceData.width,
                            height: sliceData.height,
                            values: sliceValues,
                            pointX: slicePointX,
                            pointY: slicePointY,
                            pointLabel,
                            points: promptPoints,
                            windowMin,
                            windowMax,
                            inferenceQuality,
                        });
                    }
                    selectedTotal += sliceResult.selectedIndices.length;
                    iouScore += sliceResult.iouScore;
                    encodeMs += sliceResult.timings.encodeMs;
                    decodeMs += sliceResult.timings.decodeMs;
                    if (sliceResult.embeddingCacheHit) cacheHitCount += 1;
                    changed += this.applySelectionIndicesToActiveRoi(
                        axis,
                        sliceIndex,
                        sliceData.width,
                        sliceData.height,
                        sliceResult.selectedIndices,
                    );
                }
                const avgIou = targetSlices.length > 0 ? iouScore / targetSlices.length : 0;
                const cacheSummary = cacheHitCount === targetSlices.length
                    ? 'cache'
                    : cacheHitCount === 0
                        ? 'encoded'
                        : `${cacheHitCount}/${targetSlices.length} cache`;
                this.setSmartRegionStatus(
                    `SAM2 ${promptTag} (${qualityTag}): ${selectedTotal.toLocaleString()} voxels (${cacheSummary}, ${targetSlices.length} slices, ${totalMs.toFixed(0)}ms)`,
                );
                if (changed > 0) {
                    this.invalidateActiveRoiStats();
                    this.scheduleAxisRender(axis);
                    this.scheduleActiveRoiStatsRefresh({ force: true });
                    this.updatePixelInfo();
                    this.schedule3DMaskSync({ immediate: true, render: true });
                }
                console.info(
                    `[SmartRegion] axis=${axis} slice=${hit.sliceIndex} selected=${selectedTotal} changed=${changed} preview=${this.smartRegionPreviewOnly} quality=${result.qualityUsed} points=${promptPoints.length} slices=${targetSlices.length} total=${totalMs.toFixed(2)}ms encode=${encodeMs.toFixed(2)}ms decode=${decodeMs.toFixed(2)}ms iou=${avgIou.toFixed(3)} cacheHits=${cacheHitCount}`,
                );
                this.refreshSmartRegionPreviewControls();
                return;
            }
            this.refreshSmartRegionPreviewControls();
            console.info(
                `[SmartRegion] axis=${axis} slice=${hit.sliceIndex} selected=${result.selectedIndices.length} changed=${changed} preview=${this.smartRegionPreviewOnly} quality=${result.qualityUsed} points=${promptPoints.length} total=${totalMs.toFixed(2)}ms encode=${result.timings.encodeMs.toFixed(2)}ms decode=${result.timings.decodeMs.toFixed(2)}ms iou=${result.iouScore.toFixed(3)} cache=${result.embeddingCacheHit ? 'hit' : 'miss'}`,
            );
        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            this.setSmartRegionStatus(`SAM2 failed: ${message}`, true);
            console.error('Smart region tool failed:', error);
        } finally {
            this.segmentationWorkerBusy = false;
            this.setActiveRoiBusy(active.id, false);
        }
    }

    private async runRegionGrowSlice(task: {
        width: number;
        height: number;
        values: Float32Array;
        seedIndex: number;
        tolerance: number;
    }): Promise<RegionGrowSliceResult> {
        if (this.gpu?.isLost) {
            this.disableRegionGrowWebGpu('WebGPU device is lost. Falling back to worker backend.');
        }
        if (this.segmentationGpuCompute && !this.segmentationGpuComputeFailed) {
            try {
                const startedAt = performance.now();
                const selected = await this.segmentationGpuCompute.runRegionGrowSlice(task);
                const elapsedMs = performance.now() - startedAt;
                if (elapsedMs > REGION_GROW_GPU_DISABLE_AFTER_MS) {
                    this.disableRegionGrowWebGpu(
                        `WebGPU region grow is too slow on this GPU (${elapsedMs.toFixed(1)}ms slice). Falling back to worker for this session.`,
                    );
                }
                return {
                    selected,
                    backend: 'webgpu',
                    elapsedMs,
                };
            } catch (error) {
                this.disableRegionGrowWebGpu('WebGPU region grow failed; falling back to worker path.', error);
            }
        }
        const startedAt = performance.now();
        const selected = await this.segmentationWorker.runRegionGrowSlice(task);
        return {
            selected,
            backend: 'worker',
            elapsedMs: performance.now() - startedAt,
        };
    }

    private disableRegionGrowWebGpu(reason: string, error?: unknown): void {
        if (!this.segmentationGpuComputeFailed) {
            console.warn(reason, error);
        }
        this.segmentationGpuComputeFailed = true;
    }

    private logRegionGrowPerf(
        axis: ViewAxis,
        targetSliceCount: number,
        changedTotal: number,
        totalElapsedMs: number,
        callStats: Record<RegionGrowBackend, RegionGrowPerfStats>,
    ): void {
        for (const backend of ['webgpu', 'worker'] as const) {
            this.regionGrowPerfTotals[backend].slices += callStats[backend].slices;
            this.regionGrowPerfTotals[backend].selected += callStats[backend].selected;
            this.regionGrowPerfTotals[backend].elapsedMs += callStats[backend].elapsedMs;
        }

        const selectedTotal = callStats.webgpu.selected + callStats.worker.selected;
        const backendsUsed = (['webgpu', 'worker'] as const)
            .filter((backend) => callStats[backend].slices > 0)
            .map((backend) => {
                const stats = callStats[backend];
                const avgMs = stats.slices > 0 ? stats.elapsedMs / stats.slices : 0;
                const voxelsPerSec = stats.elapsedMs > 0 ? (stats.selected * 1000) / stats.elapsedMs : 0;
                return `${backend}(s=${stats.slices},sel=${stats.selected},avg=${avgMs.toFixed(2)}ms,vps=${Math.round(voxelsPerSec)})`;
            })
            .join(' ');
        const backendSummary = backendsUsed || 'none';

        const sessionSummary = (['webgpu', 'worker'] as const)
            .map((backend) => {
                const total = this.regionGrowPerfTotals[backend];
                if (total.slices <= 0) return `${backend}(s=0)`;
                const avgMs = total.elapsedMs / total.slices;
                const voxelsPerSec = total.elapsedMs > 0 ? (total.selected * 1000) / total.elapsedMs : 0;
                return `${backend}(s=${total.slices},avg=${avgMs.toFixed(2)}ms,vps=${Math.round(voxelsPerSec)})`;
            })
            .join(' ');

        console.info(
            `[RegionGrowPerf] axis=${axis} slices=${targetSliceCount} selected=${selectedTotal} changed=${changedTotal} total=${totalElapsedMs.toFixed(2)}ms ${backendSummary} session=${sessionSummary}`,
        );
    }

    private undoSegmentationEdit(): void {
        if (!this.segmentationStore.canUndo()) return;
        const changed = this.segmentationStore.undo();
        if (changed <= 0) return;
        this.invalidateActiveRoiStats();
        this.mark3DMaskDirty();
        this.scheduleSliceRender();
        this.scheduleActiveRoiStatsRefresh({ force: true });
        this.updatePixelInfo();
        this.schedule3DMaskSync({ immediate: true, render: true });
    }

    private redoSegmentationEdit(): void {
        if (!this.segmentationStore.canRedo()) return;
        const changed = this.segmentationStore.redo();
        if (changed <= 0) return;
        this.invalidateActiveRoiStats();
        this.mark3DMaskDirty();
        this.scheduleSliceRender();
        this.scheduleActiveRoiStatsRefresh({ force: true });
        this.updatePixelInfo();
        this.schedule3DMaskSync({ immediate: true, render: true });
    }

    private init2DControls(): void {
        for (const axis of AXES) {
            const canvas = this.sliceCanvases[axis];
            if (!canvas) continue;

            // Track active view
            canvas.addEventListener('mousedown', (e) => {
                this.updateActionShortcutOverride(e);
                this.activeAxis = axis;
                this.syncThresholdRangeUI(this.uiState.state.segmentation);
                if (!this.volume) return;

                if (this.isMeasuringMode()) {
                    // ROI mode: start drawing rectangle
                    this.roiDragging = true;
                    this.roiAxis = axis;
                    const containerRect = this.viewportContainers[axis].getBoundingClientRect();
                    this.roiStartCSS = { x: e.clientX - containerRect.left, y: e.clientY - containerRect.top };
                    this.roiEndCSS = { ...this.roiStartCSS };
                    this.createRoiOverlay(axis);
                    return;
                }

                if (this.shouldUseBrushTool(e)) {
                    this.activePaintStrokeId = this.nextPaintStrokeId++;
                    this.segmentationDragging = true;
                    this.segmentationDragAxis = axis;
                    this.paintBrushAtClient(axis, e.clientX, e.clientY, e);
                    return;
                }
                if (this.shouldUseThresholdTool(e)) {
                    void this.runThresholdToolAtClient(axis, e.clientX, e.clientY, e);
                    return;
                }
                if (this.shouldUseRegionGrowTool(e)) {
                    void this.runRegionGrowToolAtClient(axis, e.clientX, e.clientY, e);
                    return;
                }
                if (this.shouldUseSmartRegionTool()) {
                    void this.runSmartRegionToolAtClient(axis, e.clientX, e.clientY, e);
                    return;
                }

                this.slice2DDragging = true;
                this.slice2DDidMove = false;
                this.slice2DLastX = e.clientX;
                this.slice2DLastY = e.clientY;
                this.slice2DStartX = e.clientX;
                this.slice2DStartY = e.clientY;
                this.slice2DDragAxis = axis;
            });

            // Wheel: Ctrl=zoom, plain=slice scroll
            canvas.addEventListener('wheel', (e) => {
                if (!this.volume) return;
                e.preventDefault();

                if (e.ctrlKey) {
                    // Zoom
                    const factor = e.deltaY > 0 ? 1 / 1.1 : 1.1;
                    this.setZoom(this.uiState.state.zoom * factor);
                } else {
                    // Slice scroll
                    const delta = e.deltaY > 0 ? 1 : -1;
                    const current = this.uiState.state.slices[axis];
                    const next = Math.max(0, Math.min(this.maxSlice(axis), current + delta));
                    if (next !== current) {
                        this.uiState.setSlice(axis, next);
                        this.updateCrosshairSliceAxis(axis, next);
                        this.renderSlice(axis);
                        this.updateSliceIndicators();
                        this.updatePixelInfo();
                    }
                }
            }, { passive: false });

            // Double-click: maximize/restore
            canvas.addEventListener('dblclick', () => {
                this.toggleMaximize(axis);
            });
        }

        // Global mouse handlers (single set for all canvases)
        window.addEventListener('mousemove', (e) => {
            // ROI drag
            if (this.roiDragging && this.roiAxis) {
                const containerRect = this.viewportContainers[this.roiAxis].getBoundingClientRect();
                this.roiEndCSS = { x: e.clientX - containerRect.left, y: e.clientY - containerRect.top };
                this.updateRoiOverlay();
                return;
            }

            if (this.segmentationDragging && this.segmentationDragAxis) {
                this.paintBrushAtClient(this.segmentationDragAxis, e.clientX, e.clientY, e);
                return;
            }

            if (!this.slice2DDragging || !this.volume) return;

            const dx = e.clientX - this.slice2DLastX;
            const dy = e.clientY - this.slice2DLastY;
            this.slice2DLastX = e.clientX;
            this.slice2DLastY = e.clientY;

            // Check if we've moved enough to count as a drag
            if (Math.abs(e.clientX - this.slice2DStartX) > 3 || Math.abs(e.clientY - this.slice2DStartY) > 3) {
                this.slice2DDidMove = true;
            }

            if (this.slice2DDidMove) {
                if (this.crosshairsEnabled && this.slice2DDragAxis) {
                    // Crosshair drag: continuously update crosshair position
                    const canvas = this.sliceCanvases[this.slice2DDragAxis];
                    if (canvas) {
                        const rect = canvas.getBoundingClientRect();
                        const dpr = canvas.width / canvas.clientWidth;
                        const cx = (e.clientX - rect.left) * dpr;
                        const cy = (e.clientY - rect.top) * dpr;
                        this.setCrosshairFromClick(this.slice2DDragAxis, cx, cy);
                    }
                } else {
                    // Pan: convert CSS px delta to slice px delta for each renderer
                    for (const a of AXES) {
                        const r = this.sliceRenderers[a];
                        if (!r) continue;
                        const c = this.sliceCanvases[a]!;
                        const scale = c.width / c.clientWidth;
                        const [sdx, sdy] = r.canvasDeltaToSlice(dx * scale, dy * scale);
                        r.panX += sdx;
                        r.panY += sdy;
                    }
                    this.uiState.update({ panX: dx, panY: dy }); // just to trigger statechange
                    this.scheduleSliceRender();
                }
            }
        }, { signal: this.globalAbort.signal });

        window.addEventListener('mouseup', (e) => {
            this.updateActionShortcutOverride(e);
            // ROI release
            if (this.roiDragging && this.roiAxis) {
                this.roiDragging = false;
                this.applyRoiSelection(this.roiAxis);
                this.removeRoiOverlay();
                this.setRoiMode(false);
                return;
            }

            if (this.segmentationDragging) {
                this.segmentationDragging = false;
                this.segmentationDragAxis = null;
                this.activePaintStrokeId = null;
                this.updatePixelInfo();
                this.scheduleActiveRoiStatsRefresh({ force: true });
                this.schedule3DMaskSync({ immediate: true, render: true });
                return;
            }

            if (!this.slice2DDragging) return;
            this.slice2DDragging = false;

            // Click without drag: set crosshair
            if (!this.slice2DDidMove && this.crosshairsEnabled && this.volume && this.slice2DDragAxis) {
                const canvas = this.sliceCanvases[this.slice2DDragAxis];
                if (canvas) {
                    const rect = canvas.getBoundingClientRect();
                    const dpr = canvas.width / canvas.clientWidth;
                    const cx = (e.clientX - rect.left) * dpr;
                    const cy = (e.clientY - rect.top) * dpr;
                    this.setCrosshairFromClick(this.slice2DDragAxis, cx, cy);
                }
            }

            this.slice2DDragAxis = null;
        }, { signal: this.globalAbort.signal });
    }

    // ================================================================
    // Crosshair system
    // ================================================================

    private toggleCrosshairs(): void {
        this.crosshairsEnabled = !this.crosshairsEnabled;
        this.uiState.update({ crosshairsEnabled: this.crosshairsEnabled });

        // Toggle crosshair button active state
        const btn = document.getElementById('crosshairBtn');
        if (btn) {
            btn.classList.toggle('active', this.crosshairsEnabled);
        }

        // Show/hide pixel info
        if (this.pixelInfoGroup) {
            this.pixelInfoGroup.style.display = this.crosshairsEnabled ? 'inline-flex' : 'none';
        }

        this.updateCrosshairs();
        this.renderSlices();
    }

    private setCrosshairFromClick(axis: ViewAxis, canvasX: number, canvasY: number): void {
        const renderer = this.sliceRenderers[axis];
        if (!renderer || !this.volume) return;

        const [px, py] = renderer.canvasToSlice(canvasX, canvasY);
        const [nx, ny, nz] = this.volume.dimensions;

        // Map slice pixel coords to volume coords based on axis
        switch (axis) {
            case 'xy':
                this.crosshairPos.x = Math.max(0, Math.min(nx - 1, Math.floor(px)));
                this.crosshairPos.y = Math.max(0, Math.min(ny - 1, Math.floor(py)));
                this.crosshairPos.z = this.uiState.state.slices.xy;
                break;
            case 'xz':
                this.crosshairPos.x = Math.max(0, Math.min(nx - 1, Math.floor(px)));
                this.crosshairPos.z = Math.max(0, Math.min(nz - 1, Math.floor(py)));
                this.crosshairPos.y = this.uiState.state.slices.xz;
                break;
            case 'yz':
                this.crosshairPos.y = Math.max(0, Math.min(ny - 1, Math.floor(px)));
                this.crosshairPos.z = Math.max(0, Math.min(nz - 1, Math.floor(py)));
                this.crosshairPos.x = this.uiState.state.slices.yz;
                break;
        }

        // Update slice indices from crosshair position
        this.uiState.update({
            slices: {
                xy: this.crosshairPos.z,
                xz: this.crosshairPos.y,
                yz: this.crosshairPos.x,
            },
        });

        this.updateCrosshairs();
        this.updateSliceIndicators();
        this.updatePixelInfo();
        this.renderAll();
    }

    /** When a slice index changes via scroll, update the corresponding crosshair axis */
    private updateCrosshairSliceAxis(axis: ViewAxis, index: number): void {
        switch (axis) {
            case 'xy': this.crosshairPos.z = index; break;
            case 'xz': this.crosshairPos.y = index; break;
            case 'yz': this.crosshairPos.x = index; break;
        }
    }

    /** Push crosshair coords to all slice renderers */
    private updateCrosshairs(): void {
        const { x, y, z } = this.crosshairPos;
        const enabled = this.crosshairsEnabled;

        const rXY = this.sliceRenderers.xy;
        if (rXY) {
            rXY.crosshairX = x + 0.5;
            rXY.crosshairY = y + 0.5;
            rXY.crosshairEnabled = enabled;
        }

        const rXZ = this.sliceRenderers.xz;
        if (rXZ) {
            rXZ.crosshairX = x + 0.5;
            rXZ.crosshairY = z + 0.5;
            rXZ.crosshairEnabled = enabled;
        }

        const rYZ = this.sliceRenderers.yz;
        if (rYZ) {
            rYZ.crosshairX = y + 0.5;
            rYZ.crosshairY = z + 0.5;
            rYZ.crosshairEnabled = enabled;
        }
    }

    private updatePixelInfo(): void {
        if (!this.volume || !this.pixelInfoEl) return;
        const { x, y, z } = this.crosshairPos;
        const val = this.volume.getValue(x, y, z);
        const maskVal = this.maskVolume ? this.maskVolume.getVoxel(x, y, z) : 0;
        const intVal = val == null ? '-' : Math.round(val).toString();
        this.pixelInfoEl.textContent = `X: ${x}, Y: ${y}, Z: ${z} = ${intVal} | Mask: ${maskVal}`;
    }

    // ================================================================
    // ROI selection
    // ================================================================

    private setRoiMode(on: boolean): void {
        this.setAppMode(on ? AppMode.Measuring : AppMode.Viewing);

        if (!on) {
            this.roiDragging = false;
            this.removeRoiOverlay();
        }
    }

    private toggleRoiMode(): void {
        this.setRoiMode(!this.isMeasuringMode());
    }

    private createRoiOverlay(axis: ViewAxis): void {
        this.removeRoiOverlay();
        const overlay = document.createElement('div');
        overlay.className = 'roi-overlay';
        this.viewportContainers[axis].appendChild(overlay);
        this.roiOverlay = overlay;
    }

    private updateRoiOverlay(): void {
        if (!this.roiOverlay) return;
        const x1 = Math.min(this.roiStartCSS.x, this.roiEndCSS.x);
        const y1 = Math.min(this.roiStartCSS.y, this.roiEndCSS.y);
        const x2 = Math.max(this.roiStartCSS.x, this.roiEndCSS.x);
        const y2 = Math.max(this.roiStartCSS.y, this.roiEndCSS.y);

        this.roiOverlay.style.left = `${x1}px`;
        this.roiOverlay.style.top = `${y1}px`;
        this.roiOverlay.style.width = `${x2 - x1}px`;
        this.roiOverlay.style.height = `${y2 - y1}px`;
    }

    private removeRoiOverlay(): void {
        this.roiOverlay?.remove();
        this.roiOverlay = null;
    }

    private applyRoiSelection(axis: ViewAxis): void {
        const renderer = this.sliceRenderers[axis];
        if (!renderer || !this.volume) return;

        const canvas = this.sliceCanvases[axis]!;
        const dpr = canvas.width / canvas.clientWidth;

        // Convert CSS corners to canvas backing pixels, then to slice pixels
        const [sx1, sy1] = renderer.canvasToSlice(this.roiStartCSS.x * dpr, this.roiStartCSS.y * dpr);
        const [sx2, sy2] = renderer.canvasToSlice(this.roiEndCSS.x * dpr, this.roiEndCSS.y * dpr);

        const [sw, sh] = renderer.getSliceDimensions();
        const minSX = Math.max(0, Math.min(Math.floor(Math.min(sx1, sx2)), sw - 1));
        const maxSX = Math.max(0, Math.min(Math.floor(Math.max(sx1, sx2)), sw - 1));
        const minSY = Math.max(0, Math.min(Math.floor(Math.min(sy1, sy2)), sh - 1));
        const maxSY = Math.max(0, Math.min(Math.floor(Math.max(sy1, sy2)), sh - 1));

        if (maxSX <= minSX || maxSY <= minSY) return;

        // Get the current slice data and compute min/max in region
        const sliceIndex = this.uiState.state.slices[axis];
        const slice = this.volume.getSlice(axis, sliceIndex);
        const data = slice.data;
        const w = slice.width;

        let roiMin = Infinity;
        let roiMax = -Infinity;
        for (let py = minSY; py <= maxSY; py++) {
            for (let px = minSX; px <= maxSX; px++) {
                const val = data[py * w + px];
                if (val < roiMin) roiMin = val;
                if (val > roiMax) roiMax = val;
            }
        }

        if (roiMin >= roiMax) return;

        this.applyWindow(roiMin, roiMax);
    }

    // ================================================================
    // Zoom / Pan
    // ================================================================

    private setZoom(zoom: number): void {
        zoom = Math.max(0.1, Math.min(10.0, zoom));
        this.uiState.setZoom(zoom);
        this.syncZoomPan();
        this.scheduleSliceRender();
        this.zoomLevelEl.textContent = `${Math.round(zoom * 100)}%`;
    }

    /** Push current zoom to all slice renderers */
    private syncZoomPan(): void {
        const zoom = this.uiState.state.zoom;
        for (const axis of AXES) {
            const r = this.sliceRenderers[axis];
            if (r) r.zoom = zoom;
        }
    }

    // ================================================================
    // Histogram
    // ================================================================

    private async computeHistogram(volume: VolumeData): Promise<void> {
        const bins = new Array(256).fill(0);
        const data = volume.data;
        const min = volume.min;
        const range = volume.max - volume.min;
        if (range <= 0) {
            this.histogramBins = bins;
            return;
        }

        const CHUNK = 1_000_000;
        for (let start = 0; start < data.length; start += CHUNK) {
            const end = Math.min(start + CHUNK, data.length);
            for (let i = start; i < end; i++) {
                const bin = Math.floor(((data[i] - min) / range) * 255);
                bins[Math.max(0, Math.min(255, bin))]++;
            }
            if (start + CHUNK < data.length) {
                await new Promise<void>(r => setTimeout(r, 0));
                // Volume may have changed while yielded
                if (this.volume !== volume) return;
            }
        }
        this.histogramBins = bins;
    }

    private drawHistogram(): void {
        const canvas = this.histogramCanvas;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx || this.histogramBins.length === 0) return;

        // Scale backing store to DPR for crisp rendering
        const dpr = window.devicePixelRatio || 1;
        const cssW = canvas.clientWidth;
        const cssH = canvas.clientHeight;
        if (cssW > 0 && cssH > 0) {
            canvas.width = Math.floor(cssW * dpr);
            canvas.height = Math.floor(cssH * dpr);
        }

        const w = canvas.width;
        const h = canvas.height;
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, 0, w, h);

        if (!this.volume) return;

        // Find max for normalization (log scale)
        let maxLog = 0;
        for (const b of this.histogramBins) {
            const l = Math.log(b + 1);
            if (l > maxLog) maxLog = l;
        }
        if (maxLog === 0) return;

        const volMin = this.volume.min;
        const volRange = this.volume.max - volMin;
        const barWidth = w / 256;

        for (let i = 0; i < 256; i++) {
            const val = Math.log(this.histogramBins[i] + 1) / maxLog;
            const barHeight = val * h;

            // Is this bin in the active window?
            const binValue = volMin + (i / 255) * volRange;
            const inRange = binValue >= this.displayWindowMin && binValue <= this.displayWindowMax;

            ctx.fillStyle = inRange ? '#4a9eff' : '#3a3a3a';
            ctx.fillRect(i * barWidth, h - barHeight, Math.ceil(barWidth), barHeight);
        }

        // Draw window boundary lines
        if (volRange > 0) {
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1;

            const minX = ((this.displayWindowMin - volMin) / volRange) * w;
            ctx.beginPath();
            ctx.moveTo(minX, 0);
            ctx.lineTo(minX, h);
            ctx.stroke();

            const maxX = ((this.displayWindowMax - volMin) / volRange) * w;
            ctx.beginPath();
            ctx.moveTo(maxX, 0);
            ctx.lineTo(maxX, h);
            ctx.stroke();
        }
    }

    private initHistogramHandles(): void {
        const handleMin = document.getElementById('handleMin');
        const handleMax = document.getElementById('handleMax');
        if (!handleMin || !handleMax) return;

        const startDrag = (which: 'min' | 'max') => (e: MouseEvent) => {
            e.preventDefault();
            this.histDragging = which;
        };

        handleMin.addEventListener('mousedown', startDrag('min'));
        handleMax.addEventListener('mousedown', startDrag('max'));

        window.addEventListener('mousemove', (e) => {
            if (!this.histDragging || !this.volume) return;

            const container = this.histogramCanvas.parentElement;
            if (!container) return;
            const rect = container.getBoundingClientRect();
            const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            const volMin = this.volume.min;
            const volRange = this.volume.max - volMin;
            const value = volMin + ratio * volRange;

            if (this.histDragging === 'min') {
                this.displayWindowMin = Math.min(value, this.displayWindowMax - volRange * 0.004);
            } else {
                this.displayWindowMax = Math.max(value, this.displayWindowMin + volRange * 0.004);
            }

            this.applyWindow(this.displayWindowMin, this.displayWindowMax);
        }, { signal: this.globalAbort.signal });

        window.addEventListener('mouseup', () => {
            this.histDragging = null;
        }, { signal: this.globalAbort.signal });
    }

    private applyWindow(min: number, max: number): void {
        this.displayWindowMin = min;
        this.displayWindowMax = max;

        for (const axis of AXES) {
            this.sliceRenderers[axis]?.setWindow(min, max);
        }
        this.mipRenderer?.setWindow(min, max);

        this.drawHistogram();
        this.updateHandlePositions();
        this.updateHistogramLabels();
        this.scheduleRender();
    }

    private updateHandlePositions(): void {
        if (!this.volume) return;
        const handleMin = document.getElementById('handleMin');
        const handleMax = document.getElementById('handleMax');
        if (!handleMin || !handleMax) return;

        const volMin = this.volume.min;
        const volRange = this.volume.max - volMin;
        if (volRange <= 0) return;

        const container = this.histogramCanvas.parentElement;
        if (!container) return;
        const containerWidth = container.clientWidth;

        const minPos = ((this.displayWindowMin - volMin) / volRange) * containerWidth;
        const maxPos = ((this.displayWindowMax - volMin) / volRange) * containerWidth;

        handleMin.style.left = `${minPos - 5}px`;
        handleMax.style.left = `${maxPos - 5}px`;
    }

    private updateHistogramLabels(): void {
        const minLabel = document.getElementById('histogramMin');
        const maxLabel = document.getElementById('histogramMax');
        if (minLabel) minLabel.textContent = Math.round(this.displayWindowMin).toString();
        if (maxLabel) maxLabel.textContent = Math.round(this.displayWindowMax).toString();
    }

    // ================================================================
    // View maximize (double-click)
    // ================================================================

    private toggleMaximize(viewId: string): void {
        if (this.maximizedView === viewId) {
            // Restore
            this.ct3DView.classList.remove('maximized');
            for (const el of Object.values(this.viewportContainers)) {
                el.classList.remove('hidden-viewport', 'maximized-viewport');
            }
            this.maximizedView = null;
        } else {
            // Maximize
            this.ct3DView.classList.add('maximized');
            for (const [id, el] of Object.entries(this.viewportContainers)) {
                if (id === viewId) {
                    el.classList.remove('hidden-viewport');
                    el.classList.add('maximized-viewport');
                } else {
                    el.classList.add('hidden-viewport');
                    el.classList.remove('maximized-viewport');
                }
            }
            this.maximizedView = viewId;
        }
    }

    // ================================================================
    // 3D mouse controls
    // ================================================================

    private init3DControls(): void {
        const canvas = this.canvas3D;
        let dragging = false;
        let lastX = 0;
        let lastY = 0;

        canvas.addEventListener('mousedown', (e) => {
            if (!this.mipRenderer || !this.volume) return;
            dragging = true;
            this.begin3DInteractionQuality();
            lastX = e.clientX;
            lastY = e.clientY;
            canvas.style.cursor = 'grabbing';
        });

        window.addEventListener('mousemove', (e) => {
            if (!dragging || !this.mipRenderer) return;

            const dx = e.clientX - lastX;
            const dy = e.clientY - lastY;
            lastX = e.clientX;
            lastY = e.clientY;

            // Left-right: spin turntable around Z-axis (azimuth)
            this.mipRenderer.azimuth += dx * 0.5 * DEG;
            // Up-down: continuous elevation rotation (no clamp)
            this.mipRenderer.elevation -= dy * 0.5 * DEG;

            this.mipRenderer.render();
        }, { signal: this.globalAbort.signal });

        window.addEventListener('mouseup', () => {
            if (dragging) {
                dragging = false;
                canvas.style.cursor = 'grab';
                this.end3DInteractionQuality();
            }
        }, { signal: this.globalAbort.signal });

        canvas.addEventListener('wheel', (e) => {
            if (!this.mipRenderer || !this.volume) return;
            e.preventDefault();
            this.begin3DInteractionQuality();

            const delta = e.deltaY > 0 ? -0.1 : 0.1;
            this.mipRenderer.distance = Math.max(0.3, Math.min(5.0,
                this.mipRenderer.distance + delta));

            this.mipRenderer.render();
            this.end3DInteractionQuality();
        }, { passive: false });

        // Double-click: maximize 3D view
        canvas.addEventListener('dblclick', () => {
            this.toggleMaximize('3d');
        });
    }

    // ================================================================
    // Sidebar controls
    // ================================================================

    private updateSegmentationToolRows(activeTool: SegmentationSettings['activeTool']): void {
        const actionRow = document.getElementById('segActionRow');
        const brushRow = document.getElementById('segBrushRow');
        const thresholdRow = document.getElementById('segThresholdRow');
        const growRow = document.getElementById('segGrowRow');
        const aiRow = document.getElementById('segAIRow');
        if (actionRow) actionRow.style.display = '';
        if (brushRow) brushRow.style.display = activeTool === 'brush' ? '' : 'none';
        if (thresholdRow) thresholdRow.style.display = activeTool === 'threshold' ? '' : 'none';
        if (growRow) growRow.style.display = activeTool === 'region-grow' ? '' : 'none';
        if (aiRow) aiRow.style.display = activeTool === 'smart-region' ? '' : 'none';
    }

    private updateSegmentationToolButtons(activeTool: SegmentationSettings['activeTool']): void {
        const brushBtn = document.getElementById('segToolBrushBtn');
        const thresholdBtn = document.getElementById('segToolThresholdBtn');
        const growBtn = document.getElementById('segToolGrowBtn');
        const aiBtn = document.getElementById('segToolAIBtn');
        if (brushBtn) brushBtn.classList.toggle('active', activeTool === 'brush');
        if (thresholdBtn) thresholdBtn.classList.toggle('active', activeTool === 'threshold');
        if (growBtn) growBtn.classList.toggle('active', activeTool === 'region-grow');
        if (aiBtn) aiBtn.classList.toggle('active', activeTool === 'smart-region');
    }

    private refreshSegmentationActionButtons(): void {
        const basePaintValue = this.uiState.state.segmentation.paintValue;
        const paintValue = this.actionShortcutOverride ?? basePaintValue;
        const drawBtn = document.getElementById('segBrushDrawBtn');
        const eraseBtn = document.getElementById('segBrushEraseBtn');
        if (drawBtn) drawBtn.classList.toggle('active', paintValue === 1);
        if (eraseBtn) eraseBtn.classList.toggle('active', paintValue === 0);
    }

    private initSidebarControls(): void {
        // 3D Resolution select
        const resolutionSelect = document.getElementById('resolution3DSelect') as HTMLSelectElement | null;
        if (resolutionSelect) {
            resolutionSelect.addEventListener('change', async () => {
                await this.set3DResolution(resolutionSelect.value as 'low' | 'mid' | 'full');
                this.update3DStatusChip();
            });
        }

        // Quality select
        const qualitySelect = document.getElementById('quality3DSelect') as HTMLSelectElement | null;
        if (qualitySelect) {
            this.preferred3DQuality = qualitySelect.value as QualityPreset;
            qualitySelect.addEventListener('change', () => {
                this.preferred3DQuality = qualitySelect.value as QualityPreset;
                if (this.mipRenderer && !this.interactionQualityActive) {
                    this.mipRenderer.setQuality(this.preferred3DQuality);
                    this.mipRenderer.render();
                }
            });
        }

        // Gamma slider
        const gammaSlider = document.getElementById('gamma3DSlider') as HTMLInputElement | null;
        const gammaValue = document.getElementById('gamma3DValue');
        if (gammaSlider) {
            gammaSlider.addEventListener('input', () => {
                const val = parseFloat(gammaSlider.value);
                if (this.mipRenderer) {
                    this.mipRenderer.gamma = val;
                    this.mipRenderer.render();
                }
                if (gammaValue) gammaValue.textContent = val.toFixed(1);
            });
        }

        // Render mode select
        const renderModeSelect = document.getElementById('renderModeSelect') as HTMLSelectElement | null;
        const tfPresetGroup = document.getElementById('tfPresetGroup');
        const lightingGroup = document.getElementById('lightingGroup');
        const tfPresetSelect = document.getElementById('tfPresetSelect') as HTMLSelectElement | null;

        // Populate TF preset dropdown
        if (tfPresetSelect) {
            for (const preset of TF_PRESETS) {
                const opt = document.createElement('option');
                opt.value = preset.name;
                opt.textContent = preset.name;
                tfPresetSelect.appendChild(opt);
            }
            tfPresetSelect.addEventListener('change', () => {
                const preset = TF_PRESETS.find(p => p.name === tfPresetSelect.value);
                if (preset && this.mipRenderer) {
                    this.mipRenderer.setTransferFunction(preset);
                    this.mipRenderer.render();
                }
            });
        }

        if (renderModeSelect) {
            renderModeSelect.addEventListener('change', () => {
                const mode = parseInt(renderModeSelect.value, 10);
                if (this.mipRenderer) {
                    this.mipRenderer.setRenderMode(mode);
                    this.mipRenderer.render();
                }
                const isCompositing = mode === RENDER_MODE.Compositing;
                if (tfPresetGroup) tfPresetGroup.style.display = isCompositing ? '' : 'none';
                if (lightingGroup) lightingGroup.style.display = isCompositing ? '' : 'none';
                // Sync lighting state when switching to compositing
                if (isCompositing && this.mipRenderer && lightingToggle) {
                    this.mipRenderer.setLightingEnabled(lightingToggle.checked);
                }
            });
        }

        // Lighting toggle
        const lightingToggle = document.getElementById('lightingToggle') as HTMLInputElement | null;
        if (lightingToggle) {
            lightingToggle.addEventListener('change', () => {
                if (this.mipRenderer) {
                    this.mipRenderer.setLightingEnabled(lightingToggle.checked);
                    this.mipRenderer.render();
                }
            });
        }

        // Zoom buttons
        const zoomInBtn = document.getElementById('zoomInBtn');
        const zoomOutBtn = document.getElementById('zoomOutBtn');
        if (zoomInBtn) zoomInBtn.addEventListener('click', () => this.setZoom(this.uiState.state.zoom + 0.2));
        if (zoomOutBtn) zoomOutBtn.addEventListener('click', () => this.setZoom(this.uiState.state.zoom - 0.2));

        // Crosshair button
        const crosshairBtn = document.getElementById('crosshairBtn');
        if (crosshairBtn) {
            crosshairBtn.removeAttribute('disabled');
            crosshairBtn.addEventListener('click', () => this.toggleCrosshairs());
        }

        // ROI button
        const roiBtn = document.getElementById('roiBtn');
        if (roiBtn) {
            roiBtn.removeAttribute('disabled');
            roiBtn.addEventListener('click', () => this.toggleRoiMode());
        }
        if (this.segModeBtn) {
            this.segModeBtn.removeAttribute('disabled');
            this.segModeBtn.addEventListener('click', () => this.toggleSegmentationMode());
        }
        this.syncModeButtons();

        // Segmentation controls
        const segEnableToggle = document.getElementById('segEnableToggle') as HTMLInputElement | null;
        const segVisibleToggle = document.getElementById('segVisibleToggle') as HTMLInputElement | null;
        const segOpacitySlider = document.getElementById('segOpacitySlider') as HTMLInputElement | null;
        const segOpacityValue = document.getElementById('segOpacityValue');
        const segToolBrushBtn = document.getElementById('segToolBrushBtn') as HTMLButtonElement | null;
        const segToolThresholdBtn = document.getElementById('segToolThresholdBtn') as HTMLButtonElement | null;
        const segToolGrowBtn = document.getElementById('segToolGrowBtn') as HTMLButtonElement | null;
        const segToolAIBtn = document.getElementById('segToolAIBtn') as HTMLButtonElement | null;
        const segBrushDrawBtn = document.getElementById('segBrushDrawBtn') as HTMLButtonElement | null;
        const segBrushEraseBtn = document.getElementById('segBrushEraseBtn') as HTMLButtonElement | null;
        const segBrushSize = document.getElementById('segBrushSize') as HTMLInputElement | null;
        const segBrushSizeValue = document.getElementById('segBrushSizeValue');
        const segThresholdMin = document.getElementById('segThresholdMin') as HTMLInputElement | null;
        const segThresholdMax = document.getElementById('segThresholdMax') as HTMLInputElement | null;
        const segThresholdRangeMin = document.getElementById('segThresholdRangeMin') as HTMLInputElement | null;
        const segThresholdRangeMax = document.getElementById('segThresholdRangeMax') as HTMLInputElement | null;
        const segGrowTolerance = document.getElementById('segGrowTolerance') as HTMLInputElement | null;
        const segSliceRadius = document.getElementById('segSliceRadius') as HTMLInputElement | null;
        const segShowOnlyActiveToggle = document.getElementById('segShowOnlyActiveToggle') as HTMLInputElement | null;
        const segExportRoiBtn = document.getElementById('segExportRoiBtn') as HTMLButtonElement | null;
        const segImportRoiBtn = document.getElementById('segImportRoiBtn') as HTMLButtonElement | null;
        const segExportSegBtn = document.getElementById('segExportSegBtn') as HTMLButtonElement | null;
        const segImportSegBtn = document.getElementById('segImportSegBtn') as HTMLButtonElement | null;
        const segPropagateBtn = document.getElementById('segPropagateBtn') as HTMLButtonElement | null;

        const seg = this.uiState.state.segmentation;
        if (segEnableToggle) segEnableToggle.checked = seg.enabled;
        if (segVisibleToggle) segVisibleToggle.checked = seg.visible;
        if (segOpacitySlider) segOpacitySlider.value = seg.overlayOpacity.toFixed(2);
        if (segOpacityValue) segOpacityValue.textContent = seg.overlayOpacity.toFixed(2);
        if (segBrushSize) segBrushSize.value = String(seg.brushRadius);
        if (segBrushSizeValue) segBrushSizeValue.textContent = String(seg.brushRadius);
        this.applySegmentationNumericInputConfig(seg, segThresholdMin, segThresholdMax, segGrowTolerance);
        this.syncThresholdRangeUI(seg);
        if (segSliceRadius) segSliceRadius.value = String(seg.regionGrowSliceRadius);
        const autoThresholdDisabled = !this.volume;
        if (segThresholdRangeMin) segThresholdRangeMin.disabled = autoThresholdDisabled;
        if (segThresholdRangeMax) segThresholdRangeMax.disabled = autoThresholdDisabled;
        if (segShowOnlyActiveToggle) segShowOnlyActiveToggle.checked = seg.showOnlyActive;
        this.updateSegmentationToolRows(seg.activeTool);
        this.updateSegmentationToolButtons(seg.activeTool);
        this.refreshSegmentationActionButtons();
        this.updateSegmentationPanelVisibility();
        this.setSmartRegionStatus(SMART_REGION_STATUS_DEFAULT);
        this.refreshSmartRegionPreviewControls();

        if (segEnableToggle) {
            segEnableToggle.addEventListener('change', () => {
                this.setSegmentationState({ enabled: segEnableToggle.checked });
            });
        }
        if (segVisibleToggle) {
            segVisibleToggle.addEventListener('change', () => {
                this.setSegmentationState({ visible: segVisibleToggle.checked });
                this.scheduleSliceRender();
            });
        }
        if (segOpacitySlider) {
            segOpacitySlider.addEventListener('input', () => {
                const value = Math.max(0, Math.min(1, parseFloat(segOpacitySlider.value)));
                this.setSegmentationState({ overlayOpacity: value });
                if (segOpacityValue) segOpacityValue.textContent = value.toFixed(2);
                this.scheduleSliceRender();
            });
        }
        if (segShowOnlyActiveToggle) {
            segShowOnlyActiveToggle.addEventListener('change', () => {
                this.setSegmentationState({ showOnlyActive: segShowOnlyActiveToggle.checked });
            });
        }
        if (segExportRoiBtn) {
            segExportRoiBtn.addEventListener('click', () => {
                void this.exportActiveRoi();
            });
        }
        if (segImportRoiBtn) {
            segImportRoiBtn.addEventListener('click', () => {
                if (!this.segRoiImportInput) return;
                this.segRoiImportInput.value = '';
                this.segRoiImportInput.click();
            });
        }
        if (segExportSegBtn) {
            segExportSegBtn.addEventListener('click', () => {
                this.exportSegmentationPackage();
            });
        }
        if (segImportSegBtn) {
            segImportSegBtn.addEventListener('click', () => {
                if (!this.segPackageImportInput) return;
                this.segPackageImportInput.value = '';
                this.segPackageImportInput.click();
            });
        }
        if (this.segRoiImportInput) {
            this.segRoiImportInput.addEventListener('change', () => {
                const file = this.segRoiImportInput?.files?.[0];
                if (!file) return;
                void this.importRoiFromFile(file);
            });
        }
        if (this.segPackageImportInput) {
            this.segPackageImportInput.addEventListener('change', () => {
                const file = this.segPackageImportInput?.files?.[0];
                if (!file) return;
                void this.importSegmentationPackageFromFile(file);
            });
        }
        if (segPropagateBtn) {
            segPropagateBtn.addEventListener('click', () => {
                console.info('Propagation workflow will be added in a later phase.');
            });
        }
        if (this.segAIClearCacheBtn) {
            this.segAIClearCacheBtn.addEventListener('click', () => {
                this.smartRegionService?.clearEmbeddingCache();
                this.setSmartRegionStatus('SAM2 embedding cache cleared.');
            });
        }
        if (this.segAIPreviewToggle) {
            this.segAIPreviewToggle.checked = this.smartRegionPreviewOnly;
            this.segAIPreviewToggle.addEventListener('change', () => {
                this.smartRegionPreviewOnly = this.segAIPreviewToggle?.checked ?? true;
                this.refreshSmartRegionPreviewControls();
                const modeLabel = this.smartRegionPreviewOnly ? 'preview mode enabled' : 'apply mode enabled';
                this.setSmartRegionStatus(`SAM2 ${modeLabel}.`);
            });
        }
        if (this.segAIApplyPreviewBtn) {
            this.segAIApplyPreviewBtn.addEventListener('click', () => {
                void (async () => {
                    const changed = await this.applySmartRegionPreview();
                    if (changed === 0 && !this.smartRegionPreview) {
                        this.setSmartRegionStatus('No SAM2 preview to apply.');
                    }
                })();
            });
        }
        if (this.segAIClearPreviewBtn) {
            this.segAIClearPreviewBtn.addEventListener('click', () => {
                if (!this.smartRegionPreview) {
                    this.setSmartRegionStatus('No SAM2 preview to clear.');
                    return;
                }
                this.clearSmartRegionPreview({ render: true, resetStatus: true });
            });
        }
        if (segToolBrushBtn) {
            segToolBrushBtn.addEventListener('click', () => {
                this.setSegmentationState({ activeTool: 'brush' });
            });
        }
        if (segToolThresholdBtn) {
            segToolThresholdBtn.addEventListener('click', () => {
                this.setSegmentationState({ activeTool: 'threshold' });
            });
        }
        if (segToolGrowBtn) {
            segToolGrowBtn.addEventListener('click', () => {
                this.setSegmentationState({ activeTool: 'region-grow' });
            });
        }
        if (segToolAIBtn) {
            segToolAIBtn.addEventListener('click', () => {
                this.setSegmentationState({ activeTool: 'smart-region' });
            });
        }
        if (segBrushDrawBtn) {
            segBrushDrawBtn.addEventListener('click', () => {
                this.setSegmentationState({ paintValue: 1 });
            });
        }
        if (segBrushEraseBtn) {
            segBrushEraseBtn.addEventListener('click', () => {
                this.setSegmentationState({ paintValue: 0 });
            });
        }
        if (segBrushSize) {
            segBrushSize.addEventListener('input', () => {
                const radius = Math.max(1, Math.min(64, parseInt(segBrushSize.value, 10) || 1));
                this.setSegmentationState({ brushRadius: radius });
                if (segBrushSizeValue) segBrushSizeValue.textContent = String(radius);
            });
        }
        if (segThresholdMin) {
            segThresholdMin.addEventListener('change', () => {
                this.setSegmentationState({ thresholdMin: parseFloat(segThresholdMin.value) || 0 });
            });
        }
        if (segThresholdMax) {
            segThresholdMax.addEventListener('change', () => {
                this.setSegmentationState({ thresholdMax: parseFloat(segThresholdMax.value) || 0 });
            });
        }
        if (segThresholdRangeMin) {
            segThresholdRangeMin.addEventListener('input', () => {
                this.onThresholdSliderInput('min');
            });
        }
        if (segThresholdRangeMax) {
            segThresholdRangeMax.addEventListener('input', () => {
                this.onThresholdSliderInput('max');
            });
        }
        if (segGrowTolerance) {
            segGrowTolerance.addEventListener('change', () => {
                this.setSegmentationState({ regionGrowTolerance: parseFloat(segGrowTolerance.value) || 0 });
            });
        }
        if (segSliceRadius) {
            const updateSliceRadius = () => {
                this.setSegmentationState({ regionGrowSliceRadius: parseInt(segSliceRadius.value, 10) || 0 });
            };
            segSliceRadius.addEventListener('input', updateSliceRadius);
            segSliceRadius.addEventListener('change', updateSliceRadius);
        }
    }

    // ================================================================
    // Floating overlays (dock/panels)
    // ================================================================

    private bindOverlayUI(): void {
        if (this.dockLogoBtn) {
            this.dockLogoBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.toggleAboutPopover();
            });
        }

        if (this.aboutPopover) {
            this.aboutPopover.addEventListener('mousedown', (e) => e.stopPropagation());
        }

        if (this.imageInfoEl) {
            this.imageInfoEl.classList.add('info-trigger');
            this.imageInfoEl.setAttribute('tabindex', '0');
            this.imageInfoEl.setAttribute('role', 'button');
            this.imageInfoEl.setAttribute('aria-haspopup', 'dialog');
            this.imageInfoEl.setAttribute('aria-expanded', 'false');
            this.imageInfoEl.setAttribute('title', 'Show technical details');
            this.imageInfoEl.addEventListener('mousedown', (e) => e.stopPropagation());
            this.imageInfoEl.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.toggleFooterInfoPanel();
            });
            this.imageInfoEl.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    this.toggleFooterInfoPanel();
                }
            });
        }

        if (this.footerInfoPanel) {
            this.footerInfoPanel.addEventListener('mousedown', (e) => e.stopPropagation());
        }

        if (this.histogramToggleBtn) {
            this.histogramToggleBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.toggleHistogramOverlay();
            });
        }

        if (this.histogramPinBtn) {
            this.histogramPinBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.toggleHistogramPin();
            });
        }

        if (this.histogramOverlay) {
            this.histogramOverlay.addEventListener('mousedown', (e) => e.stopPropagation());
        }
        if (this.segmentationOverlay) {
            this.segmentationOverlay.addEventListener('mousedown', (e) => e.stopPropagation());
        }
        if (this.segToolPalette) {
            this.segToolPalette.addEventListener('mousedown', (e) => e.stopPropagation());
        }
        if (this.segmentationPinBtn) {
            this.segmentationPinBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.setSegmentationState({ isPinned: !this.uiState.state.segmentation.isPinned });
            });
        }
        if (this.segmentationAddRoiBtn) {
            this.segmentationAddRoiBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.addRoiEntry(true);
                this.scheduleSliceRender();
            });
        }
        if (this.segmentationSettingsBtn) {
            this.segmentationSettingsBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                const advanced = document.getElementById('segAdvancedControls') as HTMLDetailsElement | null;
                if (!advanced) return;
                advanced.open = !advanced.open;
            });
        }
        if (this.sliceControls) {
            this.sliceControls.addEventListener('mousedown', (e) => e.stopPropagation());
        }

        if (this.viewport3DChip) {
            this.viewport3DChip.addEventListener('mousedown', (e) => e.stopPropagation());
            this.viewport3DChip.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.toggle3DPanel();
            });
        }

        if (this.viewport3DPanel) {
            this.viewport3DPanel.addEventListener('mousedown', (e) => e.stopPropagation());
        }

        document.addEventListener('mousedown', (e) => {
            const target = e.target as Node | null;
            if (!target) return;

            if (this.overlayState.aboutOpen &&
                this.aboutPopover &&
                !this.aboutPopover.contains(target) &&
                this.dockLogoBtn &&
                !this.dockLogoBtn.contains(target)) {
                this.setAboutOpen(false);
            }

            if (this.overlayState.histogramOpen &&
                !this.overlayState.histogramPinned &&
                this.histogramOverlay &&
                !this.histogramOverlay.contains(target) &&
                this.histogramToggleBtn &&
                !this.histogramToggleBtn.contains(target)) {
                this.setHistogramOpen(false);
            }

            if (this.overlayState.threeDPanelOpen &&
                this.viewport3DControls &&
                !this.viewport3DControls.contains(target)) {
                this.set3DPanelOpen(false);
            }

            if (this.overlayState.footerInfoOpen &&
                this.footerInfoPanel &&
                !this.footerInfoPanel.contains(target) &&
                this.imageInfoEl &&
                !this.imageInfoEl.contains(target)) {
                this.setFooterInfoOpen(false);
            }
        }, { signal: this.globalAbort.signal });

        window.addEventListener('resize', () => {
            if (this.overlayState.aboutOpen) {
                this.positionAboutPopover();
            }
            if (this.overlayState.histogramOpen) {
                this.refreshHistogramOverlay();
            }
            this.clampToolDockPosition();
            this.updateToolModePanelSide();
            this.clampHistogramPosition();
            this.clampSegmentationOverlayPosition();
            this.clampSliceOverlayPosition();
            this.update3DStatusChip();
            if (this.overlayState.footerInfoOpen) {
                this.refreshFooterInfoDetails();
            }
        }, { signal: this.globalAbort.signal });

        this.bindTopOverlayBehavior();
        this.bind3DOverlayBehavior();
        this.bindToolDockDrag();
        this.bindHistogramDrag();
        this.bindSegmentationOverlayDrag();
        this.bindSliceOverlayDrag();
        this.setAboutOpen(false);
        this.setHistogramOpen(false);
        this.set3DPanelOpen(false);
        this.setFooterInfoOpen(false);
        this.updateSegmentationPanelVisibility();
        this.update3DStatusChip();
    }

    private bindTopOverlayBehavior(): void {
        if (!this.topOverlay) return;

        const showTopOverlay = () => {
            this.topOverlay!.classList.add('reveal');
            if (this.overlayState.topOverlayTimer != null) {
                clearTimeout(this.overlayState.topOverlayTimer);
            }
            this.overlayState.topOverlayTimer = window.setTimeout(() => {
                if (this.topOverlay && !this.topOverlay.matches(':hover')) {
                    this.topOverlay.classList.remove('reveal');
                }
            }, 1500);
        };

        document.addEventListener('mousemove', (e) => {
            if (e.clientY <= 72) {
                showTopOverlay();
            }
        }, { signal: this.globalAbort.signal });

        this.topOverlay.addEventListener('mouseenter', () => {
            this.topOverlay!.classList.add('reveal');
            if (this.overlayState.topOverlayTimer != null) {
                clearTimeout(this.overlayState.topOverlayTimer);
            }
        });

        this.topOverlay.addEventListener('mouseleave', () => {
            if (this.overlayState.topOverlayTimer != null) {
                clearTimeout(this.overlayState.topOverlayTimer);
            }
            this.overlayState.topOverlayTimer = window.setTimeout(() => {
                this.topOverlay?.classList.remove('reveal');
            }, 700);
        });

        showTopOverlay();
    }

    private bind3DOverlayBehavior(): void {
        if (!this.canvas3D || !this.viewport3DControls) return;
        const viewport3DContainer = this.canvas3D.parentElement;
        if (!viewport3DContainer) return;

        const activateControls = () => {
            this.set3DControlsActive(true);
            this.schedule3DControlsFade();
        };

        viewport3DContainer.addEventListener('mouseenter', activateControls);
        viewport3DContainer.addEventListener('mousemove', activateControls);
        viewport3DContainer.addEventListener('mouseleave', () => {
            if (!this.overlayState.threeDPanelOpen) {
                this.set3DControlsActive(false);
            }
        });

        this.viewport3DControls.addEventListener('mouseenter', () => {
            this.set3DControlsActive(true);
        });

        this.viewport3DControls.addEventListener('mouseleave', () => {
            if (!this.overlayState.threeDPanelOpen) {
                this.schedule3DControlsFade();
            }
        });
    }

    private bindToolDockDrag(): void {
        if (!this.toolDock || !this.toolDockGrip) return;

        let savedLeft = NaN;
        let savedTop = NaN;
        try {
            savedLeft = parseInt(localStorage.getItem('viewerV2.toolDock.left') || '', 10);
            savedTop = parseInt(localStorage.getItem('viewerV2.toolDock.top') || '', 10);
        } catch {
            // Ignore storage read failures.
        }

        if (Number.isFinite(savedLeft) && Number.isFinite(savedTop)) {
            this.setToolDockPosition(savedLeft, savedTop, false);
        } else {
            this.clampToolDockPosition();
        }

        const onPointerMove = (e: PointerEvent) => {
            if (!this.overlayState.toolDockDragging) return;
            e.preventDefault();
            const nextLeft = this.overlayState.toolDockStartLeft + (e.clientX - this.overlayState.toolDockStartX);
            const nextTop = this.overlayState.toolDockStartTop + (e.clientY - this.overlayState.toolDockStartY);
            this.setToolDockPosition(nextLeft, nextTop, false);
        };

        const onPointerUp = () => {
            if (!this.overlayState.toolDockDragging || !this.toolDock) return;
            this.overlayState.toolDockDragging = false;
            this.toolDock.classList.remove('dragging');
            this.persistToolDockPosition();
            window.removeEventListener('pointermove', onPointerMove);
            window.removeEventListener('pointerup', onPointerUp);
            window.removeEventListener('pointercancel', onPointerUp);
        };

        this.toolDockGrip.addEventListener('pointerdown', (e) => {
            if (e.button !== 0 || !this.toolDock) return;
            e.preventDefault();
            e.stopPropagation();

            const containerRect = this.dropZoneEl.getBoundingClientRect();
            const dockRect = this.toolDock.getBoundingClientRect();
            this.overlayState.toolDockDragging = true;
            this.overlayState.toolDockStartX = e.clientX;
            this.overlayState.toolDockStartY = e.clientY;
            this.overlayState.toolDockStartLeft = dockRect.left - containerRect.left;
            this.overlayState.toolDockStartTop = dockRect.top - containerRect.top;

            this.toolDock.classList.add('dragging');
            window.addEventListener('pointermove', onPointerMove);
            window.addEventListener('pointerup', onPointerUp);
            window.addEventListener('pointercancel', onPointerUp);
        });
    }

    private setToolDockPosition(left: number, top: number, persist = true): void {
        if (!this.toolDock) return;

        const containerRect = this.dropZoneEl.getBoundingClientRect();
        const dockRect = this.toolDock.getBoundingClientRect();
        const maxLeft = Math.max(8, containerRect.width - dockRect.width - 8);
        const maxTop = Math.max(56, containerRect.height - dockRect.height - 8);

        const clampedLeft = Math.max(8, Math.min(maxLeft, left));
        const clampedTop = Math.max(56, Math.min(maxTop, top));

        this.toolDock.style.left = `${Math.round(clampedLeft)}px`;
        this.toolDock.style.top = `${Math.round(clampedTop)}px`;
        this.updateToolModePanelSide();

        if (this.overlayState.aboutOpen) {
            this.positionAboutPopover();
        }
        if (persist) {
            this.persistToolDockPosition();
        }
    }

    private clampToolDockPosition(): void {
        if (!this.toolDock) return;
        const left = parseFloat(this.toolDock.style.left || '16');
        const top = parseFloat(this.toolDock.style.top || '80');
        this.setToolDockPosition(left, top, false);
    }

    private persistToolDockPosition(): void {
        if (!this.toolDock) return;
        try {
            localStorage.setItem('viewerV2.toolDock.left', `${Math.round(parseFloat(this.toolDock.style.left || '16'))}`);
            localStorage.setItem('viewerV2.toolDock.top', `${Math.round(parseFloat(this.toolDock.style.top || '80'))}`);
        } catch {
            // Persistence is optional.
        }
    }

    private bindHistogramDrag(): void {
        if (!this.histogramOverlay || !this.histogramGrip) return;

        let savedLeft = NaN;
        let savedTop = NaN;
        try {
            savedLeft = parseInt(localStorage.getItem('viewerV2.histogram.left') || '', 10);
            savedTop = parseInt(localStorage.getItem('viewerV2.histogram.top') || '', 10);
        } catch {
            // Ignore storage read failures.
        }

        if (Number.isFinite(savedLeft) && Number.isFinite(savedTop)) {
            this.setHistogramOverlayPosition(savedLeft, savedTop, false);
        }

        const onPointerMove = (e: PointerEvent) => {
            if (!this.overlayState.histogramDragging) return;
            e.preventDefault();
            const nextLeft = this.overlayState.histogramStartLeft + (e.clientX - this.overlayState.histogramStartX);
            const nextTop = this.overlayState.histogramStartTop + (e.clientY - this.overlayState.histogramStartY);
            this.setHistogramOverlayPosition(nextLeft, nextTop, false);
        };

        const onPointerUp = () => {
            if (!this.overlayState.histogramDragging || !this.histogramOverlay) return;
            this.overlayState.histogramDragging = false;
            this.histogramOverlay.classList.remove('dragging');
            this.persistHistogramOverlayPosition();
            window.removeEventListener('pointermove', onPointerMove);
            window.removeEventListener('pointerup', onPointerUp);
            window.removeEventListener('pointercancel', onPointerUp);
        };

        this.histogramGrip.addEventListener('pointerdown', (e) => {
            if (e.button !== 0 || !this.histogramOverlay) return;
            e.preventDefault();
            e.stopPropagation();

            const containerRect = this.dropZoneEl.getBoundingClientRect();
            const overlayRect = this.histogramOverlay.getBoundingClientRect();
            this.overlayState.histogramDragging = true;
            this.overlayState.histogramStartX = e.clientX;
            this.overlayState.histogramStartY = e.clientY;
            this.overlayState.histogramStartLeft = overlayRect.left - containerRect.left;
            this.overlayState.histogramStartTop = overlayRect.top - containerRect.top;

            this.histogramOverlay.classList.add('dragging');
            window.addEventListener('pointermove', onPointerMove);
            window.addEventListener('pointerup', onPointerUp);
            window.addEventListener('pointercancel', onPointerUp);
        });
    }

    private setHistogramOverlayPosition(left: number, top: number, persist = true): void {
        if (!this.histogramOverlay) return;

        const containerRect = this.dropZoneEl.getBoundingClientRect();
        const panelWidth = this.histogramOverlay.offsetWidth || 260;
        const panelHeight = this.histogramOverlay.offsetHeight || 170;
        const maxLeft = Math.max(8, containerRect.width - panelWidth - 8);
        const maxTop = Math.max(56, containerRect.height - panelHeight - 8);

        const clampedLeft = Math.max(8, Math.min(maxLeft, left));
        const clampedTop = Math.max(56, Math.min(maxTop, top));

        this.histogramOverlay.style.left = `${Math.round(clampedLeft)}px`;
        this.histogramOverlay.style.top = `${Math.round(clampedTop)}px`;
        this.histogramOverlay.style.right = 'auto';

        if (persist) {
            this.persistHistogramOverlayPosition();
        }
    }

    private clampHistogramPosition(): void {
        if (!this.histogramOverlay || !this.histogramOverlay.style.left) return;
        const left = parseFloat(this.histogramOverlay.style.left || '0');
        const top = parseFloat(this.histogramOverlay.style.top || '80');
        this.setHistogramOverlayPosition(left, top, false);
    }

    private persistHistogramOverlayPosition(): void {
        if (!this.histogramOverlay || !this.histogramOverlay.style.left) return;
        try {
            localStorage.setItem('viewerV2.histogram.left', `${Math.round(parseFloat(this.histogramOverlay.style.left || '0'))}`);
            localStorage.setItem('viewerV2.histogram.top', `${Math.round(parseFloat(this.histogramOverlay.style.top || '80'))}`);
        } catch {
            // Persistence is optional.
        }
    }

    private bindSegmentationOverlayDrag(): void {
        if (!this.segmentationOverlay || !this.segmentationGrip) return;

        let savedLeft = NaN;
        let savedTop = NaN;
        try {
            savedLeft = parseInt(localStorage.getItem('viewerV2.segmentation.left') || '', 10);
            savedTop = parseInt(localStorage.getItem('viewerV2.segmentation.top') || '', 10);
        } catch {
            // Ignore storage read failures.
        }

        if (Number.isFinite(savedLeft) && Number.isFinite(savedTop)) {
            this.setSegmentationOverlayPosition(savedLeft, savedTop, false);
        }

        const onPointerMove = (e: PointerEvent) => {
            if (!this.overlayState.segmentationDragging) return;
            e.preventDefault();
            const nextLeft = this.overlayState.segmentationStartLeft + (e.clientX - this.overlayState.segmentationStartX);
            const nextTop = this.overlayState.segmentationStartTop + (e.clientY - this.overlayState.segmentationStartY);
            this.setSegmentationOverlayPosition(nextLeft, nextTop, false);
        };

        const onPointerUp = () => {
            if (!this.overlayState.segmentationDragging || !this.segmentationOverlay) return;
            this.overlayState.segmentationDragging = false;
            this.segmentationOverlay.classList.remove('dragging');
            this.persistSegmentationOverlayPosition();
            window.removeEventListener('pointermove', onPointerMove);
            window.removeEventListener('pointerup', onPointerUp);
            window.removeEventListener('pointercancel', onPointerUp);
        };

        this.segmentationGrip.addEventListener('pointerdown', (e) => {
            if (e.button !== 0 || !this.segmentationOverlay) return;
            e.preventDefault();
            e.stopPropagation();

            const containerRect = this.dropZoneEl.getBoundingClientRect();
            const overlayRect = this.segmentationOverlay.getBoundingClientRect();
            this.overlayState.segmentationDragging = true;
            this.overlayState.segmentationStartX = e.clientX;
            this.overlayState.segmentationStartY = e.clientY;
            this.overlayState.segmentationStartLeft = overlayRect.left - containerRect.left;
            this.overlayState.segmentationStartTop = overlayRect.top - containerRect.top;

            this.segmentationOverlay.classList.add('dragging');
            window.addEventListener('pointermove', onPointerMove);
            window.addEventListener('pointerup', onPointerUp);
            window.addEventListener('pointercancel', onPointerUp);
        });
    }

    private setSegmentationOverlayPosition(left: number, top: number, persist = true): void {
        if (!this.segmentationOverlay) return;

        const containerRect = this.dropZoneEl.getBoundingClientRect();
        const panelWidth = this.segmentationOverlay.offsetWidth || 320;
        const panelHeight = this.segmentationOverlay.offsetHeight || 420;
        const maxLeft = Math.max(8, containerRect.width - panelWidth - 8);
        const maxTop = Math.max(56, containerRect.height - panelHeight - 8);

        const clampedLeft = Math.max(8, Math.min(maxLeft, left));
        const clampedTop = Math.max(56, Math.min(maxTop, top));

        this.segmentationOverlay.style.left = `${Math.round(clampedLeft)}px`;
        this.segmentationOverlay.style.top = `${Math.round(clampedTop)}px`;
        this.segmentationOverlay.style.right = 'auto';

        if (persist) {
            this.persistSegmentationOverlayPosition();
        }
    }

    private clampSegmentationOverlayPosition(): void {
        if (!this.segmentationOverlay || !this.segmentationOverlay.style.left) return;
        const left = parseFloat(this.segmentationOverlay.style.left || '0');
        const top = parseFloat(this.segmentationOverlay.style.top || '268');
        this.setSegmentationOverlayPosition(left, top, false);
    }

    private persistSegmentationOverlayPosition(): void {
        if (!this.segmentationOverlay || !this.segmentationOverlay.style.left) return;
        try {
            localStorage.setItem('viewerV2.segmentation.left', `${Math.round(parseFloat(this.segmentationOverlay.style.left || '0'))}`);
            localStorage.setItem('viewerV2.segmentation.top', `${Math.round(parseFloat(this.segmentationOverlay.style.top || '268'))}`);
        } catch {
            // Persistence is optional.
        }
    }

    private bindSliceOverlayDrag(): void {
        if (!this.sliceControls || !this.sliceGrip) return;

        let savedLeft = NaN;
        let savedTop = NaN;
        try {
            savedLeft = parseInt(localStorage.getItem('viewerV2.sliceOverlay.left') || '', 10);
            savedTop = parseInt(localStorage.getItem('viewerV2.sliceOverlay.top') || '', 10);
        } catch {
            // Ignore storage read failures.
        }
        if (Number.isFinite(savedLeft) && Number.isFinite(savedTop)) {
            this.setSliceOverlayPosition(savedLeft, savedTop, false);
        }

        const onPointerMove = (e: PointerEvent) => {
            if (!this.overlayState.sliceDragging) return;
            e.preventDefault();
            const nextLeft = this.overlayState.sliceStartLeft + (e.clientX - this.overlayState.sliceStartX);
            const nextTop = this.overlayState.sliceStartTop + (e.clientY - this.overlayState.sliceStartY);
            this.setSliceOverlayPosition(nextLeft, nextTop, false);
        };

        const onPointerUp = () => {
            if (!this.overlayState.sliceDragging || !this.sliceControls) return;
            this.overlayState.sliceDragging = false;
            this.sliceControls.classList.remove('dragging');
            this.persistSliceOverlayPosition();
            window.removeEventListener('pointermove', onPointerMove);
            window.removeEventListener('pointerup', onPointerUp);
            window.removeEventListener('pointercancel', onPointerUp);
        };

        this.sliceGrip.addEventListener('pointerdown', (e) => {
            if (e.button !== 0 || !this.sliceControls) return;
            e.preventDefault();
            e.stopPropagation();

            const containerRect = this.dropZoneEl.getBoundingClientRect();
            const overlayRect = this.sliceControls.getBoundingClientRect();
            this.overlayState.sliceDragging = true;
            this.overlayState.sliceStartX = e.clientX;
            this.overlayState.sliceStartY = e.clientY;
            this.overlayState.sliceStartLeft = overlayRect.left - containerRect.left;
            this.overlayState.sliceStartTop = overlayRect.top - containerRect.top;

            this.sliceControls.classList.add('dragging');
            window.addEventListener('pointermove', onPointerMove);
            window.addEventListener('pointerup', onPointerUp);
            window.addEventListener('pointercancel', onPointerUp);
        });
    }

    private setSliceOverlayPosition(left: number, top: number, persist = true): void {
        if (!this.sliceControls) return;

        const containerRect = this.dropZoneEl.getBoundingClientRect();
        const panelWidth = this.sliceControls.offsetWidth || 260;
        const panelHeight = this.sliceControls.offsetHeight || 34;
        const maxLeft = Math.max(8, containerRect.width - panelWidth - 8);
        const maxTop = Math.max(8, containerRect.height - panelHeight - 8);

        const clampedLeft = Math.max(8, Math.min(maxLeft, left));
        const clampedTop = Math.max(8, Math.min(maxTop, top));

        this.sliceControls.style.left = `${Math.round(clampedLeft)}px`;
        this.sliceControls.style.top = `${Math.round(clampedTop)}px`;
        this.sliceControls.style.bottom = 'auto';
        this.sliceControls.style.right = 'auto';

        if (persist) {
            this.persistSliceOverlayPosition();
        }
    }

    private clampSliceOverlayPosition(): void {
        if (!this.sliceControls || !this.sliceControls.style.left || !this.sliceControls.style.top) return;
        const left = parseFloat(this.sliceControls.style.left || '16');
        const top = parseFloat(this.sliceControls.style.top || '16');
        this.setSliceOverlayPosition(left, top, false);
    }

    private persistSliceOverlayPosition(): void {
        if (!this.sliceControls || !this.sliceControls.style.left) return;
        try {
            localStorage.setItem('viewerV2.sliceOverlay.left', `${Math.round(parseFloat(this.sliceControls.style.left || '16'))}`);
            localStorage.setItem('viewerV2.sliceOverlay.top', `${Math.round(parseFloat(this.sliceControls.style.top || '16'))}`);
        } catch {
            // Persistence is optional.
        }
    }

    private set3DControlsActive(active: boolean): void {
        if (!this.viewport3DControls) return;
        this.viewport3DControls.classList.toggle('active', !!active);
    }

    private schedule3DControlsFade(): void {
        if (this.overlayState.threeDControlsTimer != null) {
            clearTimeout(this.overlayState.threeDControlsTimer);
        }
        this.overlayState.threeDControlsTimer = window.setTimeout(() => {
            if (!this.overlayState.threeDPanelOpen) {
                this.set3DControlsActive(false);
            }
        }, 2000);
    }

    private set3DPanelOpen(open: boolean): void {
        this.overlayState.threeDPanelOpen = !!open;

        if (this.viewport3DControls) {
            this.viewport3DControls.classList.toggle('expanded', this.overlayState.threeDPanelOpen);
        }
        if (this.viewport3DPanel) {
            this.viewport3DPanel.setAttribute('aria-hidden', this.overlayState.threeDPanelOpen ? 'false' : 'true');
        }

        this.set3DControlsActive(true);
        if (!this.overlayState.threeDPanelOpen) {
            this.schedule3DControlsFade();
        }
    }

    private toggle3DPanel(forceOpen: boolean | null = null): void {
        const next = forceOpen === null ? !this.overlayState.threeDPanelOpen : !!forceOpen;
        this.set3DPanelOpen(next);
    }

    private setAboutOpen(open: boolean): void {
        this.overlayState.aboutOpen = !!open;

        if (this.aboutPopover) {
            this.aboutPopover.classList.toggle('open', this.overlayState.aboutOpen);
            this.aboutPopover.setAttribute('aria-hidden', this.overlayState.aboutOpen ? 'false' : 'true');
        }
        if (this.dockLogoBtn) {
            this.dockLogoBtn.classList.toggle('active', this.overlayState.aboutOpen);
        }
        if (this.overlayState.aboutOpen) {
            this.positionAboutPopover();
        }
    }

    private toggleAboutPopover(forceOpen: boolean | null = null): void {
        const next = forceOpen === null ? !this.overlayState.aboutOpen : !!forceOpen;
        this.setAboutOpen(next);
    }

    private setFooterInfoOpen(open: boolean): void {
        this.overlayState.footerInfoOpen = !!open;

        if (this.footerInfoPanel) {
            this.footerInfoPanel.classList.toggle('open', this.overlayState.footerInfoOpen);
            this.footerInfoPanel.setAttribute('aria-hidden', this.overlayState.footerInfoOpen ? 'false' : 'true');
        }
        if (this.imageInfoEl) {
            this.imageInfoEl.classList.toggle('active', this.overlayState.footerInfoOpen);
            this.imageInfoEl.setAttribute('aria-expanded', this.overlayState.footerInfoOpen ? 'true' : 'false');
        }
        if (this.overlayState.footerInfoOpen) {
            this.refreshFooterInfoDetails();
        }
    }

    private toggleFooterInfoPanel(forceOpen: boolean | null = null): void {
        const next = forceOpen === null ? !this.overlayState.footerInfoOpen : !!forceOpen;
        this.setFooterInfoOpen(next);
    }

    private refreshFooterInfoDetails(): void {
        if (!this.footerInfoGrid) return;
        const volume = this.volume;
        if (!volume) {
            this.footerInfoGrid.innerHTML = '<span class="footer-info-label">Status</span><span class="footer-info-value">No volume loaded</span>';
            return;
        }

        const info = volume.getInfo();
        const [nx, ny, nz] = info.dimensions;
        const [sx, sy, sz] = info.spacing;
        const mode = this.uiState.state.appMode;
        const modeLabel = mode === AppMode.Segmentation
            ? 'Segmentation'
            : mode === AppMode.Measuring
                ? 'Measuring'
                : 'Viewing';
        const streamingLabel = volume.isStreaming ? 'Yes' : 'No';
        const memoryLabel = `${info.memorySizeMB} MB`;
        const bytesLabel = this.formatBytes(this.estimateVolumeBytes(info.dimensions, info.dataType));
        const voxelCount = nx * ny * nz;
        const spacingLabel = `${this.formatNumeric(sx, 3)} x ${this.formatNumeric(sy, 3)} x ${this.formatNumeric(sz, 3)}`;

        this.footerInfoGrid.innerHTML = `
            <span class="footer-info-label">Dimensions</span><span class="footer-info-value">${nx} x ${ny} x ${nz}</span>
            <span class="footer-info-label">Voxel Count</span><span class="footer-info-value">${this.formatNumeric(voxelCount, 0)}</span>
            <span class="footer-info-label">Spacing</span><span class="footer-info-value">${spacingLabel}</span>
            <span class="footer-info-label">Type</span><span class="footer-info-value">${info.dataType}</span>
            <span class="footer-info-label">Memory</span><span class="footer-info-value">${memoryLabel} (${bytesLabel})</span>
            <span class="footer-info-label">Intensity</span><span class="footer-info-value">${this.formatNumeric(volume.min, 2)} to ${this.formatNumeric(volume.max, 2)}</span>
            <span class="footer-info-label">Streaming</span><span class="footer-info-value">${streamingLabel}</span>
            <span class="footer-info-label">Mode</span><span class="footer-info-value">${modeLabel}</span>
        `;
    }

    private formatBytes(bytes: number): string {
        if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
        const units = ['B', 'KB', 'MB', 'GB', 'TB'];
        let size = bytes;
        let unitIndex = 0;
        while (size >= 1024 && unitIndex < units.length - 1) {
            size /= 1024;
            unitIndex++;
        }
        const digits = size >= 100 || unitIndex === 0 ? 0 : size >= 10 ? 1 : 2;
        return `${size.toFixed(digits)} ${units[unitIndex]}`;
    }

    private estimateVolumeBytes(dimensions: [number, number, number], dataType: string): number {
        const [nx, ny, nz] = dimensions;
        const bytesPerVoxel = this.bytesPerVoxel(dataType);
        return nx * ny * nz * bytesPerVoxel;
    }

    private bytesPerVoxel(dataType: string): number {
        switch (dataType.toLowerCase()) {
            case 'uint8':
            case 'int8':
                return 1;
            case 'uint16':
            case 'int16':
                return 2;
            case 'uint32':
            case 'int32':
            case 'float32':
                return 4;
            case 'float64':
                return 8;
            default:
                return 1;
        }
    }

    private positionAboutPopover(): void {
        if (!this.aboutPopover || !this.toolDock) return;

        const containerRect = this.dropZoneEl.getBoundingClientRect();
        const dockRect = this.toolDock.getBoundingClientRect();
        const panelWidth = this.aboutPopover.offsetWidth || 320;
        const panelHeight = this.aboutPopover.offsetHeight || 164;
        const gap = 10;

        let left = (dockRect.right - containerRect.left) + gap;
        let top = dockRect.top - containerRect.top;

        const minLeft = 8;
        const maxLeft = Math.max(8, containerRect.width - panelWidth - 8);
        const minTop = 8;
        const maxTop = Math.max(8, containerRect.height - panelHeight - 8);

        if (left > maxLeft) {
            left = (dockRect.left - containerRect.left) - panelWidth - gap;
        }

        const clampedLeft = Math.max(minLeft, Math.min(maxLeft, left));
        const clampedTop = Math.max(minTop, Math.min(maxTop, top));

        this.aboutPopover.style.left = `${Math.round(clampedLeft)}px`;
        this.aboutPopover.style.top = `${Math.round(clampedTop)}px`;
        this.aboutPopover.style.right = 'auto';
    }

    private setHistogramOpen(open: boolean): void {
        this.overlayState.histogramOpen = !!open;

        if (this.histogramOverlay) {
            this.histogramOverlay.classList.toggle('open', this.overlayState.histogramOpen);
            this.histogramOverlay.setAttribute('aria-hidden', this.overlayState.histogramOpen ? 'false' : 'true');
        }
        if (this.histogramToggleBtn) {
            this.histogramToggleBtn.classList.toggle('active', this.overlayState.histogramOpen);
        }

        if (!this.overlayState.histogramOpen) {
            this.overlayState.histogramPinned = false;
        }
        this.updateHistogramPinUI();

        if (this.overlayState.histogramOpen) {
            this.clampHistogramPosition();
            this.refreshHistogramOverlay();
        }
    }

    private toggleHistogramOverlay(forceOpen: boolean | null = null): void {
        const next = forceOpen === null ? !this.overlayState.histogramOpen : !!forceOpen;
        this.setHistogramOpen(next);
    }

    private toggleHistogramPin(): void {
        if (!this.overlayState.histogramOpen) {
            this.setHistogramOpen(true);
        }
        this.overlayState.histogramPinned = !this.overlayState.histogramPinned;
        this.updateHistogramPinUI();
    }

    private updateHistogramPinUI(): void {
        if (!this.histogramPinBtn) return;
        this.histogramPinBtn.classList.toggle('active', this.overlayState.histogramPinned);
        this.histogramPinBtn.setAttribute('title', this.overlayState.histogramPinned ? 'Unpin histogram' : 'Pin histogram');
    }

    private refreshHistogramOverlay(): void {
        requestAnimationFrame(() => {
            this.drawHistogram();
            this.updateHandlePositions();
            this.updateHistogramLabels();
        });
    }

    private closeTransientOverlays(): boolean {
        let closed = false;

        if (this.overlayState.aboutOpen) {
            this.setAboutOpen(false);
            closed = true;
        }
        if (this.overlayState.histogramOpen && !this.overlayState.histogramPinned) {
            this.setHistogramOpen(false);
            closed = true;
        }
        if (this.overlayState.threeDPanelOpen) {
            this.set3DPanelOpen(false);
            closed = true;
        }
        if (this.overlayState.footerInfoOpen) {
            this.setFooterInfoOpen(false);
            closed = true;
        }

        return closed;
    }

    private update3DStatusChip(): void {
        if (!this.resolutionChipText) return;

        let label = '--';
        let dimsText = '--';
        const select = document.getElementById('resolution3DSelect') as HTMLSelectElement | null;
        if (select) {
            const selected = select.options[select.selectedIndex];
            if (selected) {
                const text = (selected.textContent || '').trim();
                const match = text.match(/^([^(]+)\(([^)]+)\)$/);
                if (match) {
                    label = match[1].trim();
                    dimsText = match[2].trim();
                } else if (text) {
                    label = text;
                }
            }
        }

        if (dimsText === '--' && this.volume) {
            const [nx, ny, nz] = this.volume.dimensions;
            dimsText = `${nx}x${ny}x${nz}`;
        }

        this.resolutionChipText.textContent = `Resolution: ${label} - ${dimsText}`;
        if (this.viewport3DChip) {
            const lowOrMid = label.toLowerCase() === 'low' || label.toLowerCase() === 'mid';
            this.viewport3DChip.classList.toggle('alert', lowOrMid);
        }
    }

    // ================================================================
    // Header controls + keyboard
    // ================================================================

    private initControls(): void {
        const openBtn = document.getElementById('openBtn')!;
        const resetBtn = document.getElementById('resetBtn')!;
        const rotateBtn = document.getElementById('rotateBtn');
        const fullscreenBtn = document.getElementById('fullscreenBtn')!;

        openBtn.addEventListener('click', () => this.filePicker?.open());
        resetBtn.addEventListener('click', () => this.resetView());
        if (rotateBtn) rotateBtn.addEventListener('click', () => this.rotateAllViews90());
        fullscreenBtn.addEventListener('click', () => this.toggleFullscreen());

        document.addEventListener('keydown', (e) => {
            this.updateActionShortcutOverride(e);
            const tag = (e.target as HTMLElement).tagName;
            if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
            const keyLower = e.key.toLowerCase();
            const accel = e.ctrlKey || e.metaKey;
            if (accel && this.isSegmentationMode() && keyLower === 'z') {
                e.preventDefault();
                if (e.shiftKey) {
                    this.redoSegmentationEdit();
                } else {
                    this.undoSegmentationEdit();
                }
                return;
            }
            if (accel && this.isSegmentationMode() && keyLower === 'y') {
                e.preventDefault();
                this.redoSegmentationEdit();
                return;
            }

            switch (e.key) {
                case 'o':
                    if (!e.ctrlKey && !e.metaKey) { e.preventDefault(); this.filePicker?.open(); }
                    break;
                case 'r':
                    if (!e.ctrlKey && !e.metaKey) { e.preventDefault(); this.resetView(); }
                    break;
                case 'f':
                    if (!e.ctrlKey && !e.metaKey) { e.preventDefault(); this.toggleFullscreen(); }
                    break;
                case 't':
                    if (!e.ctrlKey && !e.metaKey) { e.preventDefault(); this.rotateAllViews90(); }
                    break;
                case 'c':
                    if (!e.ctrlKey && !e.metaKey) { e.preventDefault(); this.toggleCrosshairs(); }
                    break;
                case 'h':
                    if (!e.ctrlKey && !e.metaKey) { e.preventDefault(); this.toggleHistogramOverlay(); }
                    break;
                case 's':
                    if (!e.ctrlKey && !e.metaKey) { e.preventDefault(); this.toggleSegmentationMode(); }
                    break;
                case '+':
                case '=':
                    e.preventDefault();
                    this.setZoom(this.uiState.state.zoom + 0.2);
                    break;
                case '-':
                    e.preventDefault();
                    this.setZoom(this.uiState.state.zoom - 0.2);
                    break;
                case 'ArrowLeft':
                    if (this.volume) {
                        e.preventDefault();
                        this.navigateSlice(this.activeAxis, -1);
                    }
                    break;
                case 'ArrowRight':
                    if (this.volume) {
                        e.preventDefault();
                        this.navigateSlice(this.activeAxis, 1);
                    }
                    break;
                case 'ArrowUp':
                    if (this.volume) {
                        e.preventDefault();
                        this.navigateSlice(this.activeAxis, -10);
                    }
                    break;
                case 'ArrowDown':
                    if (this.volume) {
                        e.preventDefault();
                        this.navigateSlice(this.activeAxis, 10);
                    }
                    break;
                case 'Home':
                    if (this.volume) {
                        e.preventDefault();
                        this.navigateSliceAbsolute(this.activeAxis, 0);
                    }
                    break;
                case 'End':
                    if (this.volume) {
                        e.preventDefault();
                        this.navigateSliceAbsolute(this.activeAxis, this.maxSlice(this.activeAxis));
                    }
                    break;
                case 'Escape':
                    {
                        let handled = false;
                        if (this.closeTransientOverlays()) {
                            handled = true;
                        }
                        if (this.isMeasuringMode()) {
                            this.setRoiMode(false);
                            handled = true;
                        } else if (this.isSegmentationMode()) {
                            this.setAppMode(AppMode.Viewing);
                            handled = true;
                        } else if (this.maximizedView) {
                            this.toggleMaximize(this.maximizedView);
                            handled = true;
                        }
                        if (handled) {
                            e.preventDefault();
                        }
                    }
                    break;
            }
        }, { signal: this.globalAbort.signal });
        document.addEventListener('keyup', (e) => {
            this.updateActionShortcutOverride(e);
        }, { signal: this.globalAbort.signal });
        window.addEventListener('blur', () => {
            this.updateActionShortcutOverride();
        }, { signal: this.globalAbort.signal });
        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState !== 'visible') {
                this.updateActionShortcutOverride();
            }
        }, { signal: this.globalAbort.signal });
    }

    private navigateSlice(axis: ViewAxis, delta: number): void {
        const current = this.uiState.state.slices[axis];
        const next = Math.max(0, Math.min(this.maxSlice(axis), current + delta));
        if (next !== current) {
            this.uiState.setSlice(axis, next);
            this.updateCrosshairSliceAxis(axis, next);
            this.renderSlice(axis);
            this.updateSliceIndicators();
            this.updatePixelInfo();
        }
    }

    private navigateSliceAbsolute(axis: ViewAxis, index: number): void {
        const clamped = Math.max(0, Math.min(this.maxSlice(axis), index));
        this.uiState.setSlice(axis, clamped);
        this.updateCrosshairSliceAxis(axis, clamped);
        this.renderSlice(axis);
        this.updateSliceIndicators();
        this.updatePixelInfo();
    }

    // ================================================================
    // Reset
    // ================================================================

    private resetView(): void {
        if (this.volume) {
            const [nx, ny, nz] = this.volume.dimensions;

            // Reset zoom/pan
            this.uiState.update({
                zoom: 1.0,
                panX: 0,
                panY: 0,
                slices: {
                    xy: Math.floor(nz / 2),
                    xz: Math.floor(ny / 2),
                    yz: Math.floor(nx / 2),
                },
            });

            // Reset pan on renderers
            for (const axis of AXES) {
                const r = this.sliceRenderers[axis];
                if (r) {
                    r.panX = 0;
                    r.panY = 0;
                    r.zoom = 1.0;
                    r.setRotationQuarter(0);
                }
            }
            this.viewRotationQuarter = 0;

            // Reset crosshair
            this.crosshairPos = {
                x: Math.floor(nx / 2),
                y: Math.floor(ny / 2),
                z: Math.floor(nz / 2),
            };

            // Reset window
            this.displayWindowMin = this.volume.min;
            this.displayWindowMax = this.volume.max;
            for (const axis of AXES) {
                this.sliceRenderers[axis]?.setWindow(this.volume.min, this.volume.max);
            }

            // Reset 3D
            if (this.mipRenderer) {
                this.mipRenderer.resetCamera();
                this.mipRenderer.setWindow(this.volume.min, this.volume.max);
                this.mipRenderer.gamma = 1.0;
                this.mipRenderer.setRenderMode(RENDER_MODE.MIP);
                this.mipRenderer.setLightingEnabled(true);
            }

            // Reset gamma slider
            const gammaSlider = document.getElementById('gamma3DSlider') as HTMLInputElement | null;
            const gammaValue = document.getElementById('gamma3DValue');
            if (gammaSlider) gammaSlider.value = '1';
            if (gammaValue) gammaValue.textContent = '1.0';

            // Reset render mode dropdown
            const renderModeSelect = document.getElementById('renderModeSelect') as HTMLSelectElement | null;
            if (renderModeSelect) renderModeSelect.value = '0';
            const tfPresetGroup = document.getElementById('tfPresetGroup');
            const lightingGroup = document.getElementById('lightingGroup');
            const lightingToggle = document.getElementById('lightingToggle') as HTMLInputElement | null;
            if (tfPresetGroup) tfPresetGroup.style.display = 'none';
            if (lightingGroup) lightingGroup.style.display = 'none';
            if (lightingToggle) lightingToggle.checked = true;

            // Reset histogram
            this.drawHistogram();
            this.updateHandlePositions();
            this.updateHistogramLabels();

            // Un-maximize
            if (this.maximizedView) {
                this.toggleMaximize(this.maximizedView);
            }

            this.updateCrosshairs();
            this.updateSliceIndicators();
            this.updatePixelInfo();
            this.renderAll();
        } else {
            this.uiState.update({
                zoom: 1.0,
                panX: 0,
                panY: 0,
                slices: { xy: 0, xz: 0, yz: 0 },
            });
        }
        this.zoomLevelEl.textContent = '100%';
    }

    private toggleFullscreen(): void {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen();
        } else {
            document.exitFullscreen();
        }
    }

    private rotateAllViews90(): void {
        this.viewRotationQuarter = (this.viewRotationQuarter + 1) % 4;

        for (const axis of AXES) {
            this.sliceRenderers[axis]?.setRotationQuarter(this.viewRotationQuarter);
        }

        if (this.mipRenderer) {
            this.mipRenderer.roll += 90 * DEG;
        }

        this.renderAll();
    }

    private begin3DInteractionQuality(): void {
        if (!this.mipRenderer) return;
        if (this.interactionQualityTimer != null) {
            clearTimeout(this.interactionQualityTimer);
            this.interactionQualityTimer = null;
        }
        if (!this.interactionQualityActive && this.preferred3DQuality !== 'low') {
            this.interactionQualityActive = true;
            this.mipRenderer.setQuality('low');
        }
    }

    private end3DInteractionQuality(): void {
        if (!this.mipRenderer) return;
        if (this.interactionQualityTimer != null) {
            clearTimeout(this.interactionQualityTimer);
        }
        this.interactionQualityTimer = window.setTimeout(() => {
            this.interactionQualityTimer = null;
            if (!this.mipRenderer) return;
            if (this.interactionQualityActive) {
                this.interactionQualityActive = false;
                this.mipRenderer.setQuality(this.preferred3DQuality);
                this.mipRenderer.render();
            }
        }, 180);
    }

    // ================================================================
    // Resize
    // ================================================================

    private initResizeObservers(): void {
        const containers = document.querySelectorAll('.viewport-container');
        const observer = new ResizeObserver((entries) => {
            for (const entry of entries) {
                const canvas = entry.target.querySelector('canvas');
                if (canvas) {
                    const { width, height } = entry.contentRect;
                    if (canvas === this.canvas3D) {
                        // 3D canvas: use DPR up to 2x, max 2048px, square aspect
                        const dpr = Math.min(window.devicePixelRatio || 1, 2);
                        const size = Math.min(width, height);
                        const targetSize = Math.max(1, Math.min(Math.floor(size * dpr), 2048));
                        canvas.width = targetSize;
                        canvas.height = targetSize;
                    } else {
                        const dpr = window.devicePixelRatio || 1;
                        canvas.width = Math.max(1, Math.floor(width * dpr));
                        canvas.height = Math.max(1, Math.floor(height * dpr));
                    }
                }
            }
            this.renderAll();
        });

        for (const container of containers) {
            observer.observe(container);
        }

        // Re-draw histogram and re-position handles on resize
        const histContainer = this.histogramCanvas?.parentElement;
        if (histContainer) {
            new ResizeObserver(() => {
                this.drawHistogram();
                this.updateHandlePositions();
            }).observe(histContainer);
        }
    }

    private showGPUError(): void {
        const errorDiv = document.getElementById('webgpu-error');
        const container = document.querySelector('.container') as HTMLElement;
        if (errorDiv) errorDiv.style.display = 'flex';
        if (container) container.style.display = 'none';
    }
}
