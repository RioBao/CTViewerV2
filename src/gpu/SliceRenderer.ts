import type { WebGPUContext } from './WebGPUContext.js';
import type { SliceData } from '../types.js';
import shaderSource from './shaders/slice.wgsl?raw';

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
    private bindGroup: GPUBindGroup | null = null;
    private bindGroupLayout: GPUBindGroupLayout;

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

    constructor(gpu: WebGPUContext, canvas: HTMLCanvasElement) {
        this.gpu = gpu;
        this.canvas = canvas;
        this.ctx = gpu.configureCanvas(canvas);

        const device = gpu.device;
        const shaderModule = device.createShaderModule({ code: shaderSource });

        // 12 floats × 4 bytes = 48 bytes (padded to 48)
        this.uniformBuffer = device.createBuffer({
            size: 48,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
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

    /** Upload new slice data to the GPU */
    updateSlice(slice: SliceData): void {
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

        // Recreate bind group with new buffer
        this.bindGroup = device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: this.sliceBuffer } },
            ],
        });

        this.hasData = true;
    }

    /** Set window/level range */
    setWindow(min: number, max: number): void {
        this.windowMin = min;
        this.windowMax = max;
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
        const sx = (canvasX - cw / 2) / (fitScale * this.zoom) - this.panX + w / 2;
        const sy = (canvasY - ch / 2) / (fitScale * this.zoom) - this.panY + h / 2;
        return [sx, sy];
    }

    /** Convert canvas pixel delta to slice pixel delta (for pan) */
    canvasDeltaToSlice(dx: number, dy: number): [number, number] {
        const cw = this.canvas.width;
        const ch = this.canvas.height;
        const w = this.sliceWidth;
        const h = this.sliceHeight;
        if (w === 0 || h === 0) return [0, 0];

        const fitScale = Math.min(cw / w, ch / h);
        return [dx / (fitScale * this.zoom), dy / (fitScale * this.zoom)];
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

        // Update uniforms (12 floats)
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
        this.uniformBuffer.destroy();
        this.hasData = false;
    }
}
