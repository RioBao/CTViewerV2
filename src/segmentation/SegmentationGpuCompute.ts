import type { WebGPUContext } from '../gpu/WebGPUContext.js';
import shaderSource from './shaders/regionGrow.wgsl?raw';

interface RegionGrowGpuTask {
    width: number;
    height: number;
    values: Float32Array;
    seedIndex: number;
    tolerance: number;
}

// BFS levels processed per GPU command buffer submission.
// After each batch, the CPU reads back state.nextEnd to detect convergence.
// 128 levels/batch → at most ~ceil(maxDim/128) mapAsync round-trips.
const BATCH_SIZE = 128;

function ensurePositiveInt(value: number, name: string): number {
    if (!Number.isFinite(value) || value <= 0) {
        throw new Error(`Invalid ${name}: ${value}`);
    }
    return Math.floor(value);
}

function destroyBuffers(buffers: GPUBuffer[]): void {
    for (const buffer of buffers) {
        try { buffer.destroy(); } catch { /* ignore device-lost cleanup */ }
    }
}

/**
 * WebGPU compute backend for 2D region-growing.
 *
 * Uses iterative BFS via dispatchWorkgroupsIndirect. Each GPU dispatch is
 * bounded by the frontier size at that BFS level, so no single dispatch can
 * run long enough to trigger the Windows TDR watchdog (DXGI_ERROR_DEVICE_HUNG).
 *
 * The frontier array is monotonically append-only:
 *   frontier[0..nextEnd-1] = all selected pixels across all BFS levels.
 * After convergence the whole range is copied to a readback buffer.
 */
export class SegmentationGpuCompute {
    private readonly device: GPUDevice;
    private readonly bfsStepPipeline: GPUComputePipeline;
    private readonly prepareNextPipeline: GPUComputePipeline;
    private readonly bfsStepLayout: GPUBindGroupLayout;
    private readonly prepareNextLayout: GPUBindGroupLayout;
    private readonly maxStorageBufferBindingSize: number;

    constructor(gpu: WebGPUContext) {
        this.device = gpu.device;
        this.maxStorageBufferBindingSize = this.device.limits.maxStorageBufferBindingSize;

        const shaderModule = this.device.createShaderModule({ code: shaderSource });

        // Use auto layout per pipeline. bfs_step doesn't reference indirectArgs
        // (binding 4) so its auto layout omits it — indirectBuffer is then only
        // used as INDIRECT in the bfs_step pass, not as storage. prepare_next
        // does write indirectArgs, so its auto layout includes binding 4 as
        // storage; it uses dispatchWorkgroups(1) so no INDIRECT usage there.
        // This eliminates the "writable storage + INDIRECT in same sync scope" error.
        this.bfsStepPipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'bfs_step' },
        });
        this.prepareNextPipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'prepare_next' },
        });

        this.bfsStepLayout = this.bfsStepPipeline.getBindGroupLayout(0);
        this.prepareNextLayout = this.prepareNextPipeline.getBindGroupLayout(0);
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

        const u32Bytes = total * Uint32Array.BYTES_PER_ELEMENT;
        if (u32Bytes > this.maxStorageBufferBindingSize) {
            throw new Error(
                `Slice buffer exceeds maxStorageBufferBindingSize (${u32Bytes} > ${this.maxStorageBufferBindingSize})`,
            );
        }

        // --- Buffer allocation ---

        const valuesBuffer = this.device.createBuffer({
            size: values.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        // visited[i] = 1 → pixel has been claimed; prevents re-queuing
        const visitedBuffer = this.device.createBuffer({
            size: u32Bytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        // Monotonic frontier list; also the final output
        const frontierBuffer = this.device.createBuffer({
            size: u32Bytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        // BfsState: { levelStart u32 @0, levelEnd u32 @4, nextEnd atomic<u32> @8, _pad u32 @12 }
        const stateBuffer = this.device.createBuffer({
            size: 4 * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        // Indirect dispatch args [x, y, z]; updated each level by prepare_next
        const indirectBuffer = this.device.createBuffer({
            size: 3 * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
        });
        const paramsBuffer = this.device.createBuffer({
            size: 8 * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        // Readback for state.nextEnd (1 u32, at byte offset 8 in stateBuffer)
        const nextEndReadback = this.device.createBuffer({
            size: Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const buffers: GPUBuffer[] = [
            valuesBuffer, visitedBuffer, frontierBuffer,
            stateBuffer, indirectBuffer, paramsBuffer, nextEndReadback,
        ];

        try {
            // --- Initialise buffers ---

            this.device.queue.writeBuffer(valuesBuffer, 0, values.buffer, values.byteOffset, values.byteLength);

            // Clear visited, then mark seed as visited
            const initEncoder = this.device.createCommandEncoder();
            initEncoder.clearBuffer(visitedBuffer);
            this.device.queue.submit([initEncoder.finish()]);
            this.device.queue.writeBuffer(
                visitedBuffer,
                task.seedIndex * Uint32Array.BYTES_PER_ELEMENT,
                new Uint32Array([1]),
            );

            // frontier[0] = seedIndex
            this.device.queue.writeBuffer(frontierBuffer, 0, new Uint32Array([task.seedIndex]));

            // state = { levelStart:0, levelEnd:1, nextEnd:1, _pad:0 }
            // nextEnd starts at 1 (seed already in frontier slot 0)
            this.device.queue.writeBuffer(stateBuffer, 0, new Uint32Array([0, 1, 1, 0]));

            // First bfs_step processes 1 seed pixel → 1 workgroup
            this.device.queue.writeBuffer(indirectBuffer, 0, new Uint32Array([1, 1, 1]));

            const paramsRaw = new ArrayBuffer(32);
            const pv = new DataView(paramsRaw);
            pv.setUint32(0, width, true);
            pv.setUint32(4, height, true);
            pv.setUint32(8, total, true);
            pv.setFloat32(12, tolerance, true);
            pv.setFloat32(16, seedValue, true);
            this.device.queue.writeBuffer(paramsBuffer, 0, paramsRaw);

            // bfs_step bind group: no binding 4 (indirectBuffer not in its auto layout)
            // → indirectBuffer is only INDIRECT in the bfs_step pass, not storage.
            const bfsStepBindGroup = this.device.createBindGroup({
                layout: this.bfsStepLayout,
                entries: [
                    { binding: 0, resource: { buffer: valuesBuffer } },
                    { binding: 1, resource: { buffer: visitedBuffer } },
                    { binding: 2, resource: { buffer: frontierBuffer } },
                    { binding: 3, resource: { buffer: stateBuffer } },
                    { binding: 5, resource: { buffer: paramsBuffer } },
                ],
            });

            // prepare_next bind group: only bindings 3 (state) and 4 (indirectArgs),
            // which are the only ones prepare_next actually references in the shader.
            // → indirectBuffer is only STORAGE in the prepare_next pass, not INDIRECT.
            const prepareNextBindGroup = this.device.createBindGroup({
                layout: this.prepareNextLayout,
                entries: [
                    { binding: 3, resource: { buffer: stateBuffer } },
                    { binding: 4, resource: { buffer: indirectBuffer } },
                ],
            });

            // --- Iterative BFS ---
            //
            // Max BFS depth for an 8-connected grid ≤ max(width, height).
            // We run BATCH_SIZE levels per command buffer, then read back state.nextEnd.
            // Convergence: nextEnd didn't grow compared to start of the batch → done.

            const maxDepth = width + height; // conservative upper bound
            let prevNextEnd = 1; // seed is already in frontier
            let finalNextEnd = 1;

            for (let levelsRun = 0; levelsRun < maxDepth; levelsRun += BATCH_SIZE) {
                const levelsThisBatch = Math.min(BATCH_SIZE, maxDepth - levelsRun);
                const encoder = this.device.createCommandEncoder();

                for (let i = 0; i < levelsThisBatch; i++) {
                    const stepPass = encoder.beginComputePass();
                    stepPass.setPipeline(this.bfsStepPipeline);
                    stepPass.setBindGroup(0, bfsStepBindGroup);
                    stepPass.dispatchWorkgroupsIndirect(indirectBuffer, 0);
                    stepPass.end();

                    const prepPass = encoder.beginComputePass();
                    prepPass.setPipeline(this.prepareNextPipeline);
                    prepPass.setBindGroup(0, prepareNextBindGroup);
                    prepPass.dispatchWorkgroups(1);
                    prepPass.end();
                }

                // Copy state.nextEnd (byte offset 8) to readback buffer
                encoder.copyBufferToBuffer(stateBuffer, 8, nextEndReadback, 0, Uint32Array.BYTES_PER_ELEMENT);
                this.device.queue.submit([encoder.finish()]);

                await nextEndReadback.mapAsync(GPUMapMode.READ);
                finalNextEnd = new Uint32Array(nextEndReadback.getMappedRange())[0] ?? 0;
                nextEndReadback.unmap();

                // If nextEnd didn't grow since last batch, the frontier is empty
                if (finalNextEnd === prevNextEnd) break;
                prevNextEnd = finalNextEnd;
            }

            if (finalNextEnd <= 0) return new Uint32Array(0);

            // --- Read back selected pixels: frontier[0..finalNextEnd-1] ---

            const selectedReadback = this.device.createBuffer({
                size: finalNextEnd * Uint32Array.BYTES_PER_ELEMENT,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            });
            try {
                const copyEncoder = this.device.createCommandEncoder();
                copyEncoder.copyBufferToBuffer(
                    frontierBuffer, 0,
                    selectedReadback, 0,
                    finalNextEnd * Uint32Array.BYTES_PER_ELEMENT,
                );
                this.device.queue.submit([copyEncoder.finish()]);

                await selectedReadback.mapAsync(GPUMapMode.READ);
                const result = new Uint32Array(selectedReadback.getMappedRange()).slice();
                selectedReadback.unmap();
                return result;
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
