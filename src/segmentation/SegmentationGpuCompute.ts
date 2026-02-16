import type { WebGPUContext } from '../gpu/WebGPUContext.js';
import shaderSource from './shaders/regionGrow.wgsl?raw';

interface RegionGrowGpuTask {
    width: number;
    height: number;
    values: Float32Array;
    seedIndex: number;
    tolerance: number;
}

const WORKGROUP_SIZE = 256;
const MAX_PERSISTENT_WORKGROUPS = 256;

function ensurePositiveInt(value: number, name: string): number {
    if (!Number.isFinite(value) || value <= 0) {
        throw new Error(`Invalid ${name}: ${value}`);
    }
    return Math.floor(value);
}

function destroyBuffers(buffers: GPUBuffer[]): void {
    for (const buffer of buffers) {
        try {
            buffer.destroy();
        } catch {
            // Ignore device-lost cleanup failures.
        }
    }
}

/**
 * WebGPU compute backend for 2D region-growing.
 * Uses a persistent atomic work queue to avoid per-iteration CPU readbacks.
 */
export class SegmentationGpuCompute {
    private readonly device: GPUDevice;
    private readonly pipeline: GPUComputePipeline;
    private readonly bindGroupLayout: GPUBindGroupLayout;
    private readonly maxWorkgroupsPerDispatch: number;
    private readonly maxStorageBufferBindingSize: number;

    constructor(gpu: WebGPUContext) {
        this.device = gpu.device;
        this.maxWorkgroupsPerDispatch = this.device.limits.maxComputeWorkgroupsPerDimension;
        this.maxStorageBufferBindingSize = this.device.limits.maxStorageBufferBindingSize;

        const shaderModule = this.device.createShaderModule({ code: shaderSource });
        this.pipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'region_grow_persistent',
            },
        });
        this.bindGroupLayout = this.pipeline.getBindGroupLayout(0);
    }

    async runRegionGrowSlice(task: RegionGrowGpuTask): Promise<Uint32Array> {
        const width = ensurePositiveInt(task.width, 'width');
        const height = ensurePositiveInt(task.height, 'height');
        const total = width * height;
        const values = task.values.buffer instanceof ArrayBuffer
            ? task.values
            : new Float32Array(task.values);

        if (values.length !== total) {
            throw new Error(`Region grow value length mismatch: expected ${total}, got ${values.length}`);
        }
        if (!Number.isInteger(task.seedIndex) || task.seedIndex < 0 || task.seedIndex >= total) {
            return new Uint32Array(0);
        }

        const tolerance = Number.isFinite(task.tolerance) ? Math.max(0, task.tolerance) : 0;
        const seedValue = values[task.seedIndex];
        const seedPasses = Math.abs(values[task.seedIndex] - seedValue) <= tolerance;
        if (!seedPasses) {
            return new Uint32Array(0);
        }
        const bytes = total * Uint32Array.BYTES_PER_ELEMENT;
        if (bytes > this.maxStorageBufferBindingSize) {
            throw new Error(
                `Slice buffer exceeds maxStorageBufferBindingSize (${bytes} > ${this.maxStorageBufferBindingSize})`,
            );
        }

        const valuesBuffer = this.device.createBuffer({
            size: values.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const visitedBuffer = this.device.createBuffer({
            size: bytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const queueBuffer = this.device.createBuffer({
            size: bytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        const stateBuffer = this.device.createBuffer({
            size: 4 * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        const paramsBuffer = this.device.createBuffer({
            size: 8 * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const countReadback = this.device.createBuffer({
            size: Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const buffers = [
            valuesBuffer,
            visitedBuffer,
            queueBuffer,
            stateBuffer,
            paramsBuffer,
            countReadback,
        ];

        try {
            this.device.queue.writeBuffer(valuesBuffer, 0, values.buffer, values.byteOffset, values.byteLength);
            const initEncoder = this.device.createCommandEncoder();
            initEncoder.clearBuffer(visitedBuffer);
            this.device.queue.submit([initEncoder.finish()]);

            const seedUint = new Uint32Array([task.seedIndex >>> 0]);
            const oneUint = new Uint32Array([1]);
            this.device.queue.writeBuffer(queueBuffer, 0, seedUint);
            this.device.queue.writeBuffer(visitedBuffer, task.seedIndex * Uint32Array.BYTES_PER_ELEMENT, oneUint);
            this.device.queue.writeBuffer(stateBuffer, 0, new Uint32Array([0, 1, 0, 0]));
            const paramsRaw = new ArrayBuffer(32);
            const paramsView = new DataView(paramsRaw);
            paramsView.setUint32(0, width, true);
            paramsView.setUint32(4, height, true);
            paramsView.setUint32(8, total, true);
            paramsView.setFloat32(12, tolerance, true);
            paramsView.setFloat32(16, seedValue, true);
            this.device.queue.writeBuffer(paramsBuffer, 0, paramsRaw);

            const bindGroup = this.device.createBindGroup({
                layout: this.bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: valuesBuffer } },
                    { binding: 1, resource: { buffer: visitedBuffer } },
                    { binding: 2, resource: { buffer: queueBuffer } },
                    { binding: 3, resource: { buffer: stateBuffer } },
                    { binding: 4, resource: { buffer: paramsBuffer } },
                ],
            });

            const suggestedWorkgroups = Math.ceil(total / WORKGROUP_SIZE);
            const workgroups = Math.max(
                1,
                Math.min(this.maxWorkgroupsPerDispatch, MAX_PERSISTENT_WORKGROUPS, suggestedWorkgroups),
            );
            if (workgroups > this.maxWorkgroupsPerDispatch) {
                throw new Error(
                    `Region grow workgroups exceeded device limit (${workgroups} > ${this.maxWorkgroupsPerDispatch})`,
                );
            }

            const encoder = this.device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(workgroups);
            pass.end();
            encoder.copyBufferToBuffer(
                stateBuffer,
                Uint32Array.BYTES_PER_ELEMENT,
                countReadback,
                0,
                Uint32Array.BYTES_PER_ELEMENT,
            );
            this.device.queue.submit([encoder.finish()]);

            await countReadback.mapAsync(GPUMapMode.READ);
            const selectedCount = new Uint32Array(countReadback.getMappedRange())[0] ?? 0;
            countReadback.unmap();

            if (selectedCount <= 0) {
                return new Uint32Array(0);
            }

            const selectedReadback = this.device.createBuffer({
                size: selectedCount * Uint32Array.BYTES_PER_ELEMENT,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            });
            try {
                const selectedEncoder = this.device.createCommandEncoder();
                selectedEncoder.copyBufferToBuffer(
                    queueBuffer,
                    0,
                    selectedReadback,
                    0,
                    selectedCount * Uint32Array.BYTES_PER_ELEMENT,
                );
                this.device.queue.submit([selectedEncoder.finish()]);

                await selectedReadback.mapAsync(GPUMapMode.READ);
                const selected = new Uint32Array(selectedReadback.getMappedRange()).slice();
                selectedReadback.unmap();
                return selected;
            } finally {
                selectedReadback.destroy();
            }
        } finally {
            destroyBuffers(buffers);
        }
    }

    dispose(): void {
        // Pipelines/bind-group layouts are GC-managed.
    }
}
