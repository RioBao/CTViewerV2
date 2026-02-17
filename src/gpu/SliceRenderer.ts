import type { WebGPUContext } from './WebGPUContext.js';
import type { MaskTypedArray, SliceData } from '../types.js';
import shaderSource from './shaders/slice.wgsl?raw';

const MASK_PALETTE_SIZE = 256;
const MASK_PALETTE_STRIDE = 4;

/**
 * Renders a 2D slice to a canvas using WebGPU.
 * Supports zoom, pan, aspect-correct fitting, and crosshair overlay.
 */
export class SliceRenderer {
    private gpu: WebGPUContext;
    private ctx: GPUCanvasContext;
    private canvas: HTMLCanvasElement;
    private pipeline: GPURenderPipeline;
    private uniformBuffer: GPUBuffer;
    private sliceBuffer: GPUBuffer | null = null;
    private maskBuffer: GPUBuffer | null = null;
    private emptyMaskBuffer: GPUBuffer;
    private maskPaletteBuffer: GPUBuffer;
    private bindGroup: GPUBindGroup | null = null;
    private bindGroupLayout: GPUBindGroupLayout;
    private maskStaging = new Uint32Array(0);
    private maskPaletteStaging = new Float32Array(MASK_PALETTE_SIZE * MASK_PALETTE_STRIDE);

    private windowMin = 0;
    private windowMax = 255;
    private sliceWidth = 0;
    private sliceHeight = 0;
    private hasData = false;

    // Zoom/pan (shared across views via ViewerApp)
    zoom = 1.0;
    panX = 0;  // in slice pixel units
    panY = 0;

    // Crosshair (in slice pixel coords, -1 = disabled)
    crosshairX = -1;
    crosshairY = -1;
    crosshairEnabled = false;
    private rotationQuarter = 0; // 0,1,2,3 = 0,90,180,270 clockwise

    // Segmentation overlay
    maskOverlayEnabled = false;
    maskOverlayOpacity = 0.4;
    maskOverlayColor: [number, number, number] = [1.0, 0.0, 0.0];

    constructor(gpu: WebGPUContext, canvas: HTMLCanvasElement) {
        this.gpu = gpu;
        this.canvas = canvas;
        this.ctx = gpu.configureCanvas(canvas);

        const device = gpu.device;
        const shaderModule = device.createShaderModule({ code: shaderSource });

        // 12 floats × 4 bytes = 48 bytes (padded to 48)
        this.uniformBuffer = device.createBuffer({
            size: 96,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.emptyMaskBuffer = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.maskPaletteBuffer = device.createBuffer({
            size: this.maskPaletteStaging.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Default palette: class 0 transparent, other classes red.
        for (let i = 0; i < MASK_PALETTE_SIZE; i++) {
            const base = i * MASK_PALETTE_STRIDE;
            this.maskPaletteStaging[base] = 1.0;
            this.maskPaletteStaging[base + 1] = 0.0;
            this.maskPaletteStaging[base + 2] = 0.0;
            this.maskPaletteStaging[base + 3] = i === 0 ? 0.0 : 1.0;
        }
        device.queue.writeBuffer(
            this.maskPaletteBuffer,
            0,
            this.maskPaletteStaging.buffer,
            this.maskPaletteStaging.byteOffset,
            this.maskPaletteStaging.byteLength,
        );

        this.bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
            ],
        });

        const pipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [this.bindGroupLayout],
        });

        this.pipeline = device.createRenderPipeline({
            layout: pipelineLayout,
            vertex: {
                module: shaderModule,
                entryPoint: 'vs',
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fs',
                targets: [{ format: gpu.preferredFormat }],
            },
            primitive: { topology: 'triangle-list' },
        });
    }

    /** Upload new slice data (and optional mask) to the GPU */
    updateSlice(slice: SliceData, maskData?: MaskTypedArray | null): void {
        const device = this.gpu.device;
        const { data, width, height } = slice;

        this.sliceWidth = width;
        this.sliceHeight = height;

        // Convert to Float32 for the shader
        const f32 = data instanceof Float32Array ? data : new Float32Array(data);

        // Recreate storage buffer if size changed
        const byteLength = f32.byteLength;
        if (!this.sliceBuffer || this.sliceBuffer.size < byteLength) {
            this.sliceBuffer?.destroy();
            this.sliceBuffer = device.createBuffer({
                size: byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
        }

        device.queue.writeBuffer(this.sliceBuffer, 0, f32.buffer, f32.byteOffset, f32.byteLength);

        let maskResource = this.emptyMaskBuffer;
        if (maskData && maskData.length === width * height) {
            const maskByteLength = maskData.length * 4;
            if (!this.maskBuffer || this.maskBuffer.size < maskByteLength) {
                this.maskBuffer?.destroy();
                this.maskBuffer = device.createBuffer({
                    size: maskByteLength,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                });
            }

            if (this.maskStaging.length !== maskData.length) {
                this.maskStaging = new Uint32Array(maskData.length);
            }
            for (let i = 0; i < maskData.length; i++) {
                this.maskStaging[i] = maskData[i];
            }
            device.queue.writeBuffer(
                this.maskBuffer,
                0,
                this.maskStaging.buffer,
                this.maskStaging.byteOffset,
                this.maskStaging.byteLength,
            );
            maskResource = this.maskBuffer;
        }

        // Recreate bind group with new buffer
        this.bindGroup = device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: this.sliceBuffer } },
                { binding: 2, resource: { buffer: maskResource } },
                { binding: 3, resource: { buffer: this.maskPaletteBuffer } },
            ],
        });

        this.hasData = true;
    }

    /** Upload per-class overlay palette as [r,g,b,a] * 256 entries. */
    setMaskPalette(palette: Float32Array): void {
        const required = MASK_PALETTE_SIZE * MASK_PALETTE_STRIDE;
        if (palette.length < required) return;
        this.gpu.device.queue.writeBuffer(
            this.maskPaletteBuffer,
            0,
            palette.buffer,
            palette.byteOffset,
            required * 4,
        );
    }

    /** Set window/level range */
    setWindow(min: number, max: number): void {
        this.windowMin = min;
        this.windowMax = max;
    }

    /** Set rotation in 90deg clockwise steps */
    setRotationQuarter(quarterTurns: number): void {
        this.rotationQuarter = ((quarterTurns % 4) + 4) % 4;
    }

    /** Get current window range */
    getWindow(): [number, number] {
        return [this.windowMin, this.windowMax];
    }

    /** Convert canvas pixel coordinates to slice pixel coordinates */
    canvasToSlice(canvasX: number, canvasY: number): [number, number] {
        const cw = this.canvas.width;
        const ch = this.canvas.height;
        const w = this.sliceWidth;
        const h = this.sliceHeight;
        if (w === 0 || h === 0) return [-1, -1];

        const fitScale = Math.min(cw / w, ch / h);
        const [rx, ry] = this.rotateCanvasDelta(
            canvasX - cw / 2,
            canvasY - ch / 2,
            (4 - this.rotationQuarter) % 4,
        );
        const sx = rx / (fitScale * this.zoom) - this.panX + w / 2;
        const sy = ry / (fitScale * this.zoom) - this.panY + h / 2;
        return [sx, sy];
    }

    /** Convert slice pixel coordinates to canvas pixel coordinates */
    sliceToCanvas(sliceX: number, sliceY: number): [number, number] {
        const cw = this.canvas.width;
        const ch = this.canvas.height;
        const w = this.sliceWidth;
        const h = this.sliceHeight;
        if (w === 0 || h === 0) return [-1, -1];

        const fitScale = Math.min(cw / w, ch / h);
        const rx = (sliceX + this.panX - w / 2) * fitScale * this.zoom;
        const ry = (sliceY + this.panY - h / 2) * fitScale * this.zoom;
        const [dx, dy] = this.rotateCanvasDelta(rx, ry, this.rotationQuarter);
        return [dx + cw / 2, dy + ch / 2];
    }

    /** Convert canvas pixel delta to slice pixel delta (for pan) */
    canvasDeltaToSlice(dx: number, dy: number): [number, number] {
        const cw = this.canvas.width;
        const ch = this.canvas.height;
        const w = this.sliceWidth;
        const h = this.sliceHeight;
        if (w === 0 || h === 0) return [0, 0];

        const fitScale = Math.min(cw / w, ch / h);
        const [rdx, rdy] = this.rotateCanvasDelta(dx, dy, (4 - this.rotationQuarter) % 4);
        return [rdx / (fitScale * this.zoom), rdy / (fitScale * this.zoom)];
    }

    /** Get slice dimensions */
    getSliceDimensions(): [number, number] {
        return [this.sliceWidth, this.sliceHeight];
    }

    /** Render the current slice to the canvas */
    render(): void {
        if (!this.hasData || !this.bindGroup) return;
        if (this.gpu.isLost) return;

        const device = this.gpu.device;
        const cw = this.canvas.width;
        const ch = this.canvas.height;
        if (cw <= 0 || ch <= 0) return;

        // Update uniforms
        const uniforms = new Float32Array([
            this.windowMin,
            this.windowMax,
            this.sliceWidth,
            this.sliceHeight,
            cw,
            ch,
            this.zoom,
            this.panX,
            this.panY,
            this.crosshairX,
            this.crosshairY,
            this.crosshairEnabled ? 1.0 : 0.0,
            this.rotationQuarter,
            this.maskOverlayEnabled ? 1.0 : 0.0,
            this.maskOverlayOpacity,
            this.maskOverlayColor[0],
            this.maskOverlayColor[1],
            this.maskOverlayColor[2],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]);
        device.queue.writeBuffer(this.uniformBuffer, 0, uniforms.buffer, uniforms.byteOffset, uniforms.byteLength);

        const encoder = device.createCommandEncoder();
        const textureView = this.ctx.getCurrentTexture().createView();

        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
        });

        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.draw(3); // fullscreen triangle
        pass.end();

        device.queue.submit([encoder.finish()]);
    }

    /** Clear the canvas to black */
    clear(): void {
        if (this.gpu.isLost) return;
        if (this.canvas.width <= 0 || this.canvas.height <= 0) return;
        const encoder = this.gpu.device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: this.ctx.getCurrentTexture().createView(),
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
        });
        pass.end();
        this.gpu.device.queue.submit([encoder.finish()]);
    }

    /** Reconfigure canvas context after resize */
    resize(): void {
        this.ctx = this.gpu.configureCanvas(this.canvas);
    }

    destroy(): void {
        this.sliceBuffer?.destroy();
        this.maskBuffer?.destroy();
        this.emptyMaskBuffer.destroy();
        this.maskPaletteBuffer.destroy();
        this.uniformBuffer.destroy();
        this.hasData = false;
    }

    private rotateCanvasDelta(x: number, y: number, quarterTurns: number): [number, number] {
        switch (((quarterTurns % 4) + 4) % 4) {
            case 1: return [-y, x];
            case 2: return [-x, -y];
            case 3: return [y, -x];
            default: return [x, y];
        }
    }
}
