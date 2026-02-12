import { WebGPUContext } from '../gpu/WebGPUContext.js';
import { SliceRenderer } from '../gpu/SliceRenderer.js';
import { MIPRenderer, TF_PRESETS, RENDER_MODE } from '../gpu/MIPRenderer.js';
import { UIState } from './UIState.js';
import { FilePicker, type FileGroup } from './FilePicker.js';
import { loadVolume } from '../loaders/VolumeLoader.js';
import type { VolumeData } from '../data/VolumeData.js';
import type { StreamingVolumeData } from '../data/StreamingVolumeData.js';
import type { ViewAxis } from '../types.js';

const DEG = Math.PI / 180;
const AXES: ViewAxis[] = ['xy', 'xz', 'yz'];

/**
 * Top-level application orchestrator.
 * Manages WebGPU renderers, user interaction, histogram, and crosshairs.
 */
export class ViewerApp {
    private gpu: WebGPUContext | null = null;
    private uiState = new UIState();
    private filePicker: FilePicker | null = null;
    private volume: VolumeData | StreamingVolumeData | null = null;

    // Renderers
    private sliceRenderers: Record<ViewAxis, SliceRenderer | null> = { xy: null, xz: null, yz: null };
    private sliceCanvases: Record<ViewAxis, HTMLCanvasElement | null> = { xy: null, xz: null, yz: null };
    private mipRenderer: MIPRenderer | null = null;

    // 3D resolution management
    private current3DResolution: 'low' | 'mid' | 'full' = 'low';
    private cached3DVolumes = new Map<string, VolumeData>();

    // Crosshair state (volume voxel coords)
    private crosshairPos = { x: 0, y: 0, z: 0 };
    private crosshairsEnabled = false;

    // Histogram
    private histogramBins: number[] = [];
    private displayWindowMin = 0;
    private displayWindowMax = 255;
    private histDragging: 'min' | 'max' | null = null;

    // ROI selection
    private roiMode = false;
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

    // Render scheduling
    private renderPending = false;

    // DOM references
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

    async initialize(): Promise<void> {
        // Grab DOM references
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
        const dropZone = document.getElementById('dropZone') as HTMLElement;

        // Initialize WebGPU
        try {
            this.gpu = await WebGPUContext.create();
            this.uiState.update({ gpuAvailable: true });

            for (const axis of AXES) {
                this.sliceRenderers[axis] = new SliceRenderer(this.gpu, this.sliceCanvases[axis]!);
            }
            this.mipRenderer = new MIPRenderer(this.gpu, this.canvas3D);

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
        this.initResizeObservers();
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
        this.mipRenderer?.render();
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
        renderer.updateSlice(slice);
        renderer.render();

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

    // ================================================================
    // Volume display setup
    // ================================================================

    private displayVolume(): void {
        const volume = this.volume;
        if (!volume) return;

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
        // Streaming volumes: upload the 4x downsampled MIP volume (→ "low")
        // Standard volumes: upload full if dimensions fit 3D texture limit, else downsample
        const mipVolume = volume.getMIPVolume();
        const maxDim3D = this.gpu?.device.limits.maxTextureDimension3D ?? 0;
        const [mnx, mny, mnz] = mipVolume.dimensions;
        const dimsFit = mnx <= maxDim3D && mny <= maxDim3D && mnz <= maxDim3D;
        if (this.mipRenderer) {
            if (dimsFit) {
                this.mipRenderer.uploadVolume(mipVolume);
                this.current3DResolution = volume.isStreaming ? 'low' : 'full';
                this.update3DResolutionIndicator(mipVolume);
            } else {
                // Full data too large for 3D texture — generate 4x downsample async
                console.warn(`MIP volume (${mnx}×${mny}×${mnz}) exceeds 3D texture limit (${maxDim3D}), downsampling`);
                this.current3DResolution = 'low';
                const status = document.getElementById('resolution3DStatus');
                if (mipVolume.canEnhance3D()) {
                    if (status) status.textContent = '0%';
                    mipVolume.createDownsampledVolume(4, (pct) => {
                        if (status) status.textContent = `${pct}%`;
                    }).then((downsampled) => {
                        if (status) status.textContent = '';
                        if (downsampled && this.mipRenderer && this.volume === volume) {
                            this.mipRenderer.uploadVolume(downsampled);
                            this.cached3DVolumes.set('low', downsampled);
                            this.update3DResolutionIndicator(downsampled);
                            this.mipRenderer.render();
                        }
                    });
                }
            }
            this.mipRenderer.resetCamera();
        }

        // Compute histogram (from MIP volume, which is always in-memory)
        this.computeHistogram(mipVolume);
        this.drawHistogram();
        this.updateHandlePositions();

        // Show sidebar controls
        const sliceControls = document.getElementById('sliceControls');
        if (sliceControls) sliceControls.style.display = 'block';
        const quality3D = document.getElementById('quality3DGroup');
        if (quality3D) quality3D.style.display = 'block';

        // Sync zoom/pan/crosshair to renderers
        this.syncZoomPan();
        this.updateCrosshairs();
        this.updateSliceIndicators();
        this.updatePixelInfo();

        // Update 3D resolution dropdown options
        this.update3DResolutionOptions();

        this.renderAll();
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
        lowOption.textContent = `Low (${lowNx}×${lowNy}×${lowNz})`;
        lowOption.disabled = false;

        // Mid (2x downsample)
        const midNx = Math.ceil(nx / 2);
        const midNy = Math.ceil(ny / 2);
        const midNz = Math.ceil(nz / 2);
        const canMid = volume.canEnhance3D() && midNx <= maxDim3D && midNy <= maxDim3D && midNz <= maxDim3D;
        midOption.textContent = `Mid (${midNx}×${midNy}×${midNz})`;
        midOption.disabled = !canMid;

        // Full
        const canFull = !volume.isStreaming && nx <= maxDim3D && ny <= maxDim3D && nz <= maxDim3D;
        fullOption.textContent = `Full (${nx}×${ny}×${nz})`;
        fullOption.disabled = !canFull;

        // If current selection is now disabled, fall back
        if ((this.current3DResolution === 'mid' && midOption.disabled) ||
            (this.current3DResolution === 'full' && fullOption.disabled)) {
            this.current3DResolution = 'low';
        }
        select.value = this.current3DResolution;
    }

    /**
     * Update the 3D resolution indicator overlay showing current dimensions.
     */
    private update3DResolutionIndicator(vol: VolumeData): void {
        const indicator = document.getElementById('resolution3D');
        if (!indicator) return;
        const [nx, ny, nz] = vol.dimensions;
        indicator.textContent = `${nx}×${ny}×${nz}`;
    }

    /**
     * Switch 3D resolution mode (low/mid/full).
     * Preserves the user's current window/level settings.
     */
    private async set3DResolution(value: 'low' | 'mid' | 'full'): Promise<void> {
        if (!this.volume || !this.mipRenderer) return;

        const select = document.getElementById('resolution3DSelect') as HTMLSelectElement | null;
        const status = document.getElementById('resolution3DStatus');

        // Save current window/level so uploadVolume doesn't clobber it
        const savedMin = this.mipRenderer.displayMin;
        const savedMax = this.mipRenderer.displayMax;

        const uploadAndRender = (vol: VolumeData) => {
            this.mipRenderer!.uploadVolume(vol);
            // Restore user's window/level
            this.mipRenderer!.displayMin = savedMin;
            this.mipRenderer!.displayMax = savedMax;
            this.mipRenderer!.render();
            // Update 3D resolution indicator
            this.update3DResolutionIndicator(vol);
        };

        const clearStatus = () => {
            if (status) status.textContent = '';
        };

        try {
            if (value === 'low') {
                if (this.volume.isStreaming) {
                    // Streaming: getMIPVolume() is the 4x downsampled lowRes
                    uploadAndRender(this.volume.getMIPVolume());
                } else {
                    // Standard: generate 4x downsample on demand
                    const cached = this.cached3DVolumes.get('low');
                    if (cached) {
                        uploadAndRender(cached);
                    } else {
                        if (select) select.disabled = true;
                        if (status) status.textContent = '0%';
                        const lowVol = await (this.volume as VolumeData).createDownsampledVolume(4, (pct) => {
                            if (status) status.textContent = `${pct}%`;
                        });
                        if (select) select.disabled = false;
                        clearStatus();
                        if (lowVol) {
                            this.cached3DVolumes.set('low', lowVol);
                            uploadAndRender(lowVol);
                        } else {
                            if (select) select.value = this.current3DResolution;
                            return;
                        }
                    }
                }
                this.current3DResolution = 'low';

            } else if (value === 'mid') {
                const cached = this.cached3DVolumes.get('mid');
                if (cached) {
                    uploadAndRender(cached);
                } else {
                    if (!this.volume.canEnhance3D()) {
                        if (select) select.value = this.current3DResolution;
                        return;
                    }
                    if (select) select.disabled = true;
                    if (status) status.textContent = '0%';
                    const midVol = await this.volume.createDownsampledVolume(2, (pct) => {
                        if (status) status.textContent = `${pct}%`;
                    });
                    if (select) select.disabled = false;
                    clearStatus();
                    if (midVol) {
                        this.cached3DVolumes.set('mid', midVol);
                        uploadAndRender(midVol);
                    } else {
                        if (select) select.value = this.current3DResolution;
                        return;
                    }
                }
                this.current3DResolution = 'mid';

            } else if (value === 'full') {
                if (this.volume.isStreaming) {
                    if (select) select.value = this.current3DResolution;
                    return;
                }
                uploadAndRender(this.volume as VolumeData);
                this.current3DResolution = 'full';
            }
        } catch (error) {
            console.error('Failed to set 3D resolution:', error);
            if (select) {
                select.disabled = false;
                select.value = this.current3DResolution;
            }
            clearStatus();
        }
    }

    private updateSliceIndicators(): void {
        if (!this.volume) return;
        const [nx, ny, nz] = this.volume.dimensions;
        const s = this.uiState.state.slices;

        this.sliceIndicators.xy.textContent = `XY: ${s.xy + 1}/${nz}`;
        this.sliceIndicators.xz.textContent = `XZ: ${s.xz + 1}/${ny}`;
        this.sliceIndicators.yz.textContent = `YZ: ${s.yz + 1}/${nx}`;
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

        // Dispose previous streaming volume if any
        if (this.volume && this.volume.isStreaming) {
            this.volume.dispose();
        }

        // Reset 3D resolution cache for new volume
        this.cached3DVolumes.clear();
        this.current3DResolution = 'low';

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
                    // Dispose the streaming volume
                    if (this.volume && this.volume.isStreaming) {
                        this.volume.dispose();
                    }
                    // Reset 3D resolution cache since we have new full data
                    this.cached3DVolumes.clear();
                    this.current3DResolution = 'low';
                    this.volume = fullVolume;
                    const fullInfo = fullVolume.getInfo();
                    this.fileNameEl.textContent = name;
                    const [w, h, d] = fullInfo.dimensions;
                    this.imageInfoEl.textContent =
                        `${w}×${h}×${d} ${fullInfo.dataType} ${fullInfo.memorySizeMB}MB`;
                    this.displayVolume();
                    this.hideLoadingOverlay();
                },
            );

            this.volume = volume;

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
                `${w}×${h}×${d} ${info.dataType} ${info.memorySizeMB}MB${suffix}`;

            this.displayVolume();
            this.uiState.setFileLoaded(name);
        } catch (err) {
            console.error('Failed to load volume:', err);
            const msg = err instanceof Error ? err.message : String(err);
            this.imageInfoEl.textContent = `Error: ${msg}`;
            this.fileNameEl.textContent = name;
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

    private init2DControls(): void {
        for (const axis of AXES) {
            const canvas = this.sliceCanvases[axis];
            if (!canvas) continue;

            // Track active view
            canvas.addEventListener('mousedown', (e) => {
                this.activeAxis = axis;
                if (!this.volume) return;

                if (this.roiMode) {
                    // ROI mode: start drawing rectangle
                    this.roiDragging = true;
                    this.roiAxis = axis;
                    const containerRect = this.viewportContainers[axis].getBoundingClientRect();
                    this.roiStartCSS = { x: e.clientX - containerRect.left, y: e.clientY - containerRect.top };
                    this.roiEndCSS = { ...this.roiStartCSS };
                    this.createRoiOverlay(axis);
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
        });

        window.addEventListener('mouseup', (e) => {
            // ROI release
            if (this.roiDragging && this.roiAxis) {
                this.roiDragging = false;
                this.applyRoiSelection(this.roiAxis);
                this.removeRoiOverlay();
                this.setRoiMode(false);
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
        });
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
            this.pixelInfoGroup.style.display = this.crosshairsEnabled ? 'block' : 'none';
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
        this.pixelInfoEl.textContent = `X: ${x}, Y: ${y}, Z: ${z} = ${val ?? '—'}`;
    }

    // ================================================================
    // ROI selection
    // ================================================================

    private setRoiMode(on: boolean): void {
        this.roiMode = on;
        this.uiState.update({ roiMode: on });

        const btn = document.getElementById('roiBtn');
        if (btn) btn.classList.toggle('active', on);

        // Change cursor on all 2D canvases
        for (const axis of AXES) {
            const c = this.sliceCanvases[axis];
            if (c) c.style.cursor = on ? 'crosshair' : '';
        }

        if (!on) {
            this.roiDragging = false;
            this.removeRoiOverlay();
        }
    }

    private toggleRoiMode(): void {
        this.setRoiMode(!this.roiMode);
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

    private computeHistogram(volume: VolumeData): void {
        const bins = new Array(256).fill(0);
        const data = volume.data;
        const min = volume.min;
        const range = volume.max - volume.min;
        if (range <= 0) {
            this.histogramBins = bins;
            return;
        }

        for (let i = 0; i < data.length; i++) {
            const bin = Math.floor(((data[i] - min) / range) * 255);
            bins[Math.max(0, Math.min(255, bin))]++;
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
        });

        window.addEventListener('mouseup', () => {
            this.histDragging = null;
        });
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
        });

        window.addEventListener('mouseup', () => {
            if (dragging) {
                dragging = false;
                canvas.style.cursor = 'grab';
            }
        });

        canvas.addEventListener('wheel', (e) => {
            if (!this.mipRenderer || !this.volume) return;
            e.preventDefault();

            const delta = e.deltaY > 0 ? -0.1 : 0.1;
            this.mipRenderer.distance = Math.max(0.3, Math.min(5.0,
                this.mipRenderer.distance + delta));

            this.mipRenderer.render();
        }, { passive: false });

        // Double-click: maximize 3D view
        canvas.addEventListener('dblclick', () => {
            this.toggleMaximize('3d');
        });
    }

    // ================================================================
    // Sidebar controls
    // ================================================================

    private initSidebarControls(): void {
        // 3D Resolution select
        const resolutionSelect = document.getElementById('resolution3DSelect') as HTMLSelectElement | null;
        if (resolutionSelect) {
            resolutionSelect.addEventListener('change', async () => {
                await this.set3DResolution(resolutionSelect.value as 'low' | 'mid' | 'full');
            });
        }

        // Quality select
        const qualitySelect = document.getElementById('quality3DSelect') as HTMLSelectElement | null;
        if (qualitySelect) {
            qualitySelect.addEventListener('change', () => {
                this.mipRenderer?.setQuality(qualitySelect.value);
                this.mipRenderer?.render();
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
    }

    // ================================================================
    // Header controls + keyboard
    // ================================================================

    private initControls(): void {
        const openBtn = document.getElementById('openBtn')!;
        const resetBtn = document.getElementById('resetBtn')!;
        const fullscreenBtn = document.getElementById('fullscreenBtn')!;

        openBtn.addEventListener('click', () => this.filePicker?.open());
        resetBtn.addEventListener('click', () => this.resetView());
        fullscreenBtn.addEventListener('click', () => this.toggleFullscreen());

        document.addEventListener('keydown', (e) => {
            const tag = (e.target as HTMLElement).tagName;
            if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;

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
                case 'c':
                    if (!e.ctrlKey && !e.metaKey) { e.preventDefault(); this.toggleCrosshairs(); }
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
                    if (this.roiMode) {
                        e.preventDefault();
                        this.setRoiMode(false);
                    } else if (this.maximizedView) {
                        e.preventDefault();
                        this.toggleMaximize(this.maximizedView);
                    }
                    break;
            }
        });
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
                if (r) { r.panX = 0; r.panY = 0; r.zoom = 1.0; }
            }

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
                        // 3D canvas: use DPR up to 2×, max 2048px, square aspect
                        const dpr = Math.min(window.devicePixelRatio || 1, 2);
                        const size = Math.min(width, height);
                        const targetSize = Math.min(Math.floor(size * dpr), 2048);
                        canvas.width = targetSize;
                        canvas.height = targetSize;
                    } else {
                        const dpr = window.devicePixelRatio || 1;
                        canvas.width = Math.floor(width * dpr);
                        canvas.height = Math.floor(height * dpr);
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
