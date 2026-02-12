import type { WebGPUContext } from './WebGPUContext.js';
import type { VolumeData } from '../data/VolumeData.js';
import shaderSource from './shaders/mip.wgsl?raw';

/** Quality presets matching the old viewer */
export interface MIPQuality {
    numSteps: number;
}

export const MIP_QUALITY: Record<string, MIPQuality> = {
    low: { numSteps: 200 },
    medium: { numSteps: 800 },
    high: { numSteps: 3200 },
};

/**
 * Renders a 3D Maximum Intensity Projection using ray marching.
 * Orthographic projection with Euler angle camera control.
 */
export class MIPRenderer {
    private gpu: WebGPUContext;
    private ctx: GPUCanvasContext;
    private pipeline: GPURenderPipeline;
    private uniformBuffer: GPUBuffer;
    private volumeBuffer: GPUBuffer | null = null;
    private bindGroup: GPUBindGroup | null = null;
    private bindGroupLayout: GPUBindGroupLayout;
    private hasVolume = false;

    // Camera state
    azimuth = 30 * Math.PI / 180;
    elevation = 20 * Math.PI / 180;
    roll = 0;
    distance = 0.75;
    panX = 0;
    panY = 0;

    // Display
    displayMin = 0;
    displayMax = 1;
    gamma = 1.0;

    // Volume info
    private dimX = 0;
    private dimY = 0;
    private dimZ = 0;
    private spacingX = 1;
    private spacingY = 1;
    private spacingZ = 1;

    // Quality
    private numSteps = MIP_QUALITY.low.numSteps;

    constructor(gpu: WebGPUContext, canvas: HTMLCanvasElement) {
        this.gpu = gpu;
        this.ctx = gpu.configureCanvas(canvas);

        const device = gpu.device;
        const shaderModule = device.createShaderModule({ code: shaderSource });

        // 20 floats × 4 bytes = 80 bytes (18 used + 2 padding)
        this.uniformBuffer = device.createBuffer({
            size: 80,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
            ],
        });

        const pipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [this.bindGroupLayout],
        });

        this.pipeline = device.createRenderPipeline({
            layout: pipelineLayout,
            vertex: { module: shaderModule, entryPoint: 'vs' },
            fragment: {
                module: shaderModule,
                entryPoint: 'fs',
                targets: [{ format: gpu.preferredFormat }],
            },
            primitive: { topology: 'triangle-list' },
        });
    }

    /** Upload volume data to GPU */
    uploadVolume(volume: VolumeData): void {
        const device = this.gpu.device;
        const [nx, ny, nz] = volume.dimensions;
        const [sx, sy, sz] = volume.spacing;

        this.dimX = nx;
        this.dimY = ny;
        this.dimZ = nz;
        this.spacingX = sx;
        this.spacingY = sy;
        this.spacingZ = sz;

        this.displayMin = volume.min;
        this.displayMax = volume.max;

        // Convert to Float32
        const f32 = volume.data instanceof Float32Array
            ? volume.data
            : new Float32Array(volume.data);

        this.volumeBuffer?.destroy();
        this.volumeBuffer = device.createBuffer({
            size: f32.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.volumeBuffer, 0, f32.buffer, f32.byteOffset, f32.byteLength);

        this.bindGroup = device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: this.volumeBuffer } },
            ],
        });

        this.hasVolume = true;
    }

    /** Set quality preset */
    setQuality(name: string): void {
        const q = MIP_QUALITY[name];
        if (q) {
            this.numSteps = q.numSteps;
        }
    }

    /** Set window/level */
    setWindow(min: number, max: number): void {
        this.displayMin = min;
        this.displayMax = max;
    }

    /** Reset camera to defaults */
    resetCamera(): void {
        this.azimuth = 30 * Math.PI / 180;
        this.elevation = 20 * Math.PI / 180;
        this.roll = 0;
        this.distance = 0.75;
        this.panX = 0;
        this.panY = 0;
    }

    /** Render the MIP projection */
    render(): void {
        if (!this.hasVolume || !this.bindGroup) return;
        if (this.gpu.isLost) return;

        const device = this.gpu.device;
        const canvas = this.ctx.canvas;

        // Write uniforms (20 floats: 18 used + 2 padding)
        const data = new Float32Array([
            this.azimuth,
            this.elevation,
            this.roll,
            this.distance,
            this.panX,
            this.panY,
            this.displayMin,
            this.displayMax,
            this.gamma,
            this.dimX,
            this.dimY,
            this.dimZ,
            this.spacingX,
            this.spacingY,
            this.spacingZ,
            this.numSteps,
            canvas.width,
            canvas.height,
            0, // padding
            0, // padding
        ]);
        device.queue.writeBuffer(this.uniformBuffer, 0, data.buffer, data.byteOffset, data.byteLength);

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
        pass.draw(3);
        pass.end();

        device.queue.submit([encoder.finish()]);
    }

    /** Clear to black */
    clear(): void {
        if (this.gpu.isLost) return;
        const encoder = this.gpu.device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: this.ctx.getCurrentTexture().createView(),
                clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
        });
        pass.end();
        this.gpu.device.queue.submit([encoder.finish()]);
    }

    destroy(): void {
        this.volumeBuffer?.destroy();
        this.uniformBuffer.destroy();
        this.hasVolume = false;
    }
}
