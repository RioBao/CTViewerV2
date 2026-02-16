import type { WebGPUContext } from './WebGPUContext.js';
import type { VolumeData } from '../data/VolumeData.js';
import { TransferFunction, TF_PRESETS, type TFPreset } from './TransferFunction.js';
import shaderSource from './shaders/mip.wgsl?raw';

/** Quality presets matching the old viewer */
export interface MIPQuality {
    numSteps: number;
}

export const MIP_QUALITY: Record<string, MIPQuality> = {
    low: { numSteps: 200 },
    medium: { numSteps: 800 },
    high: { numSteps: 1600 },
};

/** Render mode constants */
export const RENDER_MODE = {
    MIP: 0,
    MinIP: 1,
    Average: 2,
    Compositing: 3,
} as const;

/** Convert Float32Array to Float16 (Uint16Array) for compact 3D upload */
function float32ToFloat16(src: Float32Array): Uint16Array {
    const dst = new Uint16Array(src.length);
    for (let i = 0; i < src.length; i++) {
        const v = src[i];
        // Handle special cases
        if (v === 0) { dst[i] = 0; continue; }
        if (!isFinite(v)) {
            dst[i] = v > 0 ? 0x7C00 : 0xFC00;
            continue;
        }
        if (isNaN(v)) { dst[i] = 0x7E00; continue; }

        const sign = v < 0 ? 0x8000 : 0;
        const abs = Math.abs(v);

        if (abs < 5.96046448e-8) {
            dst[i] = sign;
        } else if (abs < 6.103515625e-5) {
            dst[i] = sign | Math.round(abs / 5.96046448e-8);
        } else if (abs < 65504) {
            const e = Math.floor(Math.log2(abs));
            const m = abs / Math.pow(2, e) - 1;
            dst[i] = sign | ((e + 15) << 10) | Math.round(m * 1024);
        } else {
            dst[i] = sign | 0x7C00;
        }
    }
    return dst;
}

function wrapAngle(angle: number): number {
    const tau = Math.PI * 2;
    return ((angle + Math.PI) % tau + tau) % tau - Math.PI;
}

function clamp01(value: number): number {
    if (!Number.isFinite(value)) return 0;
    return Math.max(0, Math.min(1, value));
}

/**
 * Renders a 3D volume using ray marching with multiple rendering modes.
 * Supports MIP, MinIP, Average, and Compositing (DVR) with transfer function.
 * Uses hardware-filtered 3D texture and Blinn-Phong lighting.
 */
export class MIPRenderer {
    private gpu: WebGPUContext;
    private ctx: GPUCanvasContext;
    private pipeline: GPURenderPipeline;
    private uniformBuffer: GPUBuffer;
    private volumeTexture: GPUTexture | null = null;
    private volumeSampler: GPUSampler | null = null;
    private transferFunction: TransferFunction;
    private bindGroup: GPUBindGroup | null = null;
    private bindGroupLayout: GPUBindGroupLayout;
    private brickTexture: GPUTexture | null = null;
    private brickSampler: GPUSampler | null = null;
    private labelTexture: GPUTexture | null = null;
    private labelTextureDims: [number, number, number] = [0, 0, 0];
    private maskPaletteTexture: GPUTexture | null = null;
    private maskPaletteBytes = new Uint8Array(256 * 4);
    private brickGridX = 0;
    private brickGridY = 0;
    private brickGridZ = 0;
    private maskOverlayEnabled = false;
    private maskOverlayOpacity = 0.4;
    private hasVolume = false;
    private frameCount = 0;

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

    // Render mode (0=MIP, 1=MinIP, 2=Average, 3=Compositing)
    private renderMode = 0;
    private opacityScale = 1.0;

    // Lighting (Phase 2)
    private ambient = 0.2;
    private diffuse = 0.7;
    private specular = 0.3;
    private shininess = 32.0;
    private lightingEnabled = 1.0;

    constructor(gpu: WebGPUContext, canvas: HTMLCanvasElement) {
        this.gpu = gpu;
        this.ctx = gpu.configureCanvas(canvas);

        const device = gpu.device;
        const shaderModule = device.createShaderModule({ code: shaderSource });

        // 32 floats x 4 bytes = 128 bytes
        this.uniformBuffer = device.createBuffer({
            size: 128,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.transferFunction = new TransferFunction(device);

        this.bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '3d' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
                { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
                { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '3d' } },
                { binding: 6, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
                { binding: 7, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'uint', viewDimension: '3d' } },
                { binding: 8, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
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

        this.maskPaletteTexture = device.createTexture({
            size: [256, 1, 1],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        this.setMaskPalette(this.createDefaultMaskPalette());
        this.clearLabelVolume();
    }

    /** Upload volume data to GPU as a 3D texture */
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

        // Clear previous resources first so failed uploads do not leave stale textures bound.
        this.unloadVolume();

        try {
            // Always upload as r16float to keep VRAM use predictable on very large datasets.
            const f16 = float32ToFloat16(f32);
            this.volumeTexture = device.createTexture({
                size: [nx, ny, nz],
                format: 'r16float',
                dimension: '3d',
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            });
            device.queue.writeTexture(
                { texture: this.volumeTexture },
                f16.buffer,
                { offset: f16.byteOffset, bytesPerRow: nx * 2, rowsPerImage: ny },
                [nx, ny, nz],
            );

            this.volumeSampler = device.createSampler({
                magFilter: 'linear',
                minFilter: 'linear',
            });

            // Build brick map (8^3 blocks, min/max per brick)
            const BRICK = 8;
            const bgx = Math.ceil(nx / BRICK);
            const bgy = Math.ceil(ny / BRICK);
            const bgz = Math.ceil(nz / BRICK);

            const brickF32 = new Float32Array(bgx * bgy * bgz * 2);
            for (let bz = 0; bz < bgz; bz++) {
                for (let by = 0; by < bgy; by++) {
                    for (let bx = 0; bx < bgx; bx++) {
                        let bMin = Infinity;
                        let bMax = -Infinity;
                        const x0 = bx * BRICK, x1 = Math.min(x0 + BRICK, nx);
                        const y0 = by * BRICK, y1 = Math.min(y0 + BRICK, ny);
                        const z0 = bz * BRICK, z1 = Math.min(z0 + BRICK, nz);
                        for (let z = z0; z < z1; z++) {
                            for (let y = y0; y < y1; y++) {
                                const rowBase = z * nx * ny + y * nx;
                                for (let x = x0; x < x1; x++) {
                                    const v = f32[rowBase + x];
                                    if (v < bMin) bMin = v;
                                    if (v > bMax) bMax = v;
                                }
                            }
                        }
                        const idx = (bz * bgy + by) * bgx + bx;
                        brickF32[idx * 2] = bMin;
                        brickF32[idx * 2 + 1] = bMax;
                    }
                }
            }

            const brickF16 = float32ToFloat16(brickF32);
            this.brickTexture = device.createTexture({
                size: [bgx, bgy, bgz],
                format: 'rg16float',
                dimension: '3d',
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            });
            device.queue.writeTexture(
                { texture: this.brickTexture },
                brickF16.buffer,
                { offset: brickF16.byteOffset, bytesPerRow: bgx * 4, rowsPerImage: bgy },
                [bgx, bgy, bgz],
            );

            this.brickSampler = device.createSampler({
                magFilter: 'nearest',
                minFilter: 'nearest',
            });
            this.brickGridX = bgx;
            this.brickGridY = bgy;
            this.brickGridZ = bgz;

            this.rebuildBindGroup();
            this.hasVolume = true;
        } catch (error) {
            this.unloadVolume();
            throw error;
        }
    }

    /** Drop the currently uploaded 3D volume resources but keep renderer state alive. */
    unloadVolume(): void {
        this.volumeTexture?.destroy();
        this.volumeTexture = null;
        this.volumeSampler = null;
        this.brickTexture?.destroy();
        this.brickTexture = null;
        this.brickSampler = null;
        this.bindGroup = null;
        this.hasVolume = false;
        this.brickGridX = 0;
        this.brickGridY = 0;
        this.brickGridZ = 0;
        this.clearLabelVolume();
    }

    /** Rebuild bind group (after texture or TF change) */
    private rebuildBindGroup(): void {
        if (!this.volumeTexture
            || !this.volumeSampler
            || !this.brickTexture
            || !this.brickSampler
            || !this.labelTexture
            || !this.maskPaletteTexture) {
            return;
        }
        this.bindGroup = this.gpu.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: this.volumeTexture.createView() },
                { binding: 2, resource: this.volumeSampler },
                { binding: 3, resource: this.transferFunction.texture.createView() },
                { binding: 4, resource: this.transferFunction.sampler },
                { binding: 5, resource: this.brickTexture.createView() },
                { binding: 6, resource: this.brickSampler },
                { binding: 7, resource: this.labelTexture.createView() },
                { binding: 8, resource: this.maskPaletteTexture.createView() },
            ],
        });
    }

    private createDefaultMaskPalette(): Float32Array {
        const palette = new Float32Array(256 * 4);
        for (let i = 0; i < 256; i++) {
            const base = i * 4;
            palette[base] = 1.0;
            palette[base + 1] = 0.0;
            palette[base + 2] = 0.0;
            palette[base + 3] = i === 0 ? 0.0 : 1.0;
        }
        return palette;
    }

    private ensureLabelTexture(dims: [number, number, number]): void {
        const [nx, ny, nz] = dims;
        if (nx <= 0 || ny <= 0 || nz <= 0) return;
        if (this.labelTexture
            && this.labelTextureDims[0] === nx
            && this.labelTextureDims[1] === ny
            && this.labelTextureDims[2] === nz) {
            return;
        }
        this.labelTexture?.destroy();
        this.labelTexture = this.gpu.device.createTexture({
            size: [nx, ny, nz],
            format: 'r8uint',
            dimension: '3d',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        this.labelTextureDims = [nx, ny, nz];
        this.rebuildBindGroup();
    }

    getVolumeDimensions(): [number, number, number] | null {
        if (!this.hasVolume) return null;
        return [this.dimX, this.dimY, this.dimZ];
    }

    setMaskOverlay(enabled: boolean, opacity: number): void {
        this.maskOverlayEnabled = enabled;
        this.maskOverlayOpacity = clamp01(opacity);
    }

    setMaskPalette(palette: Float32Array): void {
        if (!this.maskPaletteTexture) return;
        const required = 256 * 4;
        if (palette.length < required) return;
        for (let i = 0; i < required; i++) {
            this.maskPaletteBytes[i] = Math.round(clamp01(palette[i]) * 255);
        }
        this.gpu.device.queue.writeTexture(
            { texture: this.maskPaletteTexture },
            this.maskPaletteBytes.buffer,
            { offset: this.maskPaletteBytes.byteOffset, bytesPerRow: 256 * 4, rowsPerImage: 1 },
            [256, 1, 1],
        );
    }

    uploadLabelVolume(labels: Uint8Array, dims: [number, number, number]): void {
        const [nx, ny, nz] = dims;
        if (nx <= 0 || ny <= 0 || nz <= 0) return;
        if (labels.length !== nx * ny * nz) {
            throw new Error(`Label volume length mismatch: expected ${nx * ny * nz}, got ${labels.length}`);
        }

        this.ensureLabelTexture(dims);
        if (!this.labelTexture) return;

        this.gpu.device.queue.writeTexture(
            { texture: this.labelTexture },
            labels.buffer,
            { offset: labels.byteOffset, bytesPerRow: nx, rowsPerImage: ny },
            [nx, ny, nz],
        );
    }

    clearLabelVolume(): void {
        this.ensureLabelTexture([1, 1, 1]);
        if (!this.labelTexture) return;
        const empty = new Uint8Array(1);
        this.gpu.device.queue.writeTexture(
            { texture: this.labelTexture },
            empty.buffer,
            { offset: 0, bytesPerRow: 1, rowsPerImage: 1 },
            [1, 1, 1],
        );
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

    /** Set render mode (0=MIP, 1=MinIP, 2=Average, 3=Compositing) */
    setRenderMode(mode: number): void {
        this.renderMode = mode;
    }

    /** Set transfer function from preset */
    setTransferFunction(preset: TFPreset): void {
        this.transferFunction.applyPreset(preset);
    }

    /** Set opacity scale for compositing mode */
    setOpacityScale(scale: number): void {
        this.opacityScale = scale;
    }

    /** Set lighting enabled (for compositing mode) */
    setLightingEnabled(enabled: boolean): void {
        this.lightingEnabled = enabled ? 1.0 : 0.0;
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

    /** Render the volume */
    render(): void {
        if (!this.hasVolume || !this.bindGroup) return;
        if (this.gpu.isLost) return;

        const device = this.gpu.device;
        const canvas = this.ctx.canvas;
        if (canvas.width <= 0 || canvas.height <= 0) return;

        this.azimuth = wrapAngle(this.azimuth);
        this.elevation = wrapAngle(this.elevation);
        this.roll = wrapAngle(this.roll);

        this.frameCount++;

        // Write uniforms (32 floats = 128 bytes)
        const data = new Float32Array([
            this.azimuth,       // 0
            this.elevation,     // 1
            this.roll,          // 2
            this.distance,      // 3
            this.panX,          // 4
            this.panY,          // 5
            this.displayMin,    // 6
            this.displayMax,    // 7
            this.gamma,         // 8
            this.dimX,          // 9
            this.dimY,          // 10
            this.dimZ,          // 11
            this.spacingX,      // 12
            this.spacingY,      // 13
            this.spacingZ,      // 14
            this.numSteps,      // 15
            canvas.width,       // 16
            canvas.height,      // 17
            this.frameCount,    // 18
            this.renderMode,    // 19
            this.opacityScale,  // 20
            this.ambient,       // 21
            this.diffuse,       // 22
            this.specular,      // 23
            this.shininess,     // 24
            this.lightingEnabled, // 25
            this.brickGridX,    // 26
            this.brickGridY,    // 27
            this.brickGridZ,    // 28
            this.maskOverlayEnabled ? 1 : 0, // 29
            this.maskOverlayOpacity, // 30
            0, // padding       // 31
        ]);
        device.queue.writeBuffer(this.uniformBuffer, 0, data.buffer, data.byteOffset, data.byteLength);

        const encoder = device.createCommandEncoder();
        const textureView = this.ctx.getCurrentTexture().createView();

        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0.039, g: 0.051, b: 0.075, a: 1 },
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

    /** Clear to the 3D viewport background tone */
    clear(): void {
        if (this.gpu.isLost) return;
        const canvas = this.ctx.canvas;
        if (canvas.width <= 0 || canvas.height <= 0) return;
        const encoder = this.gpu.device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: this.ctx.getCurrentTexture().createView(),
                clearValue: { r: 0.039, g: 0.051, b: 0.075, a: 1 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
        });
        pass.end();
        this.gpu.device.queue.submit([encoder.finish()]);
    }

    destroy(): void {
        this.unloadVolume();
        this.labelTexture?.destroy();
        this.labelTexture = null;
        this.maskPaletteTexture?.destroy();
        this.maskPaletteTexture = null;
        this.transferFunction.destroy();
        this.uniformBuffer.destroy();
    }
}

export { TF_PRESETS, type TFPreset };

