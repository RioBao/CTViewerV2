/**
 * Singleton manager for the WebGPU device and canvas configuration.
 */
export class WebGPUContext {
    readonly adapter: GPUAdapter;
    readonly device: GPUDevice;
    readonly preferredFormat: GPUTextureFormat;
    private _lost = false;

    private constructor(adapter: GPUAdapter, device: GPUDevice, preferredFormat: GPUTextureFormat) {
        this.adapter = adapter;
        this.device = device;
        this.preferredFormat = preferredFormat;

        this.device.lost.then((info) => {
            console.error(`WebGPU device lost: ${info.message} (reason: ${info.reason})`);
            this._lost = true;
        });
    }

    /** Whether the GPU device has been lost */
    get isLost(): boolean {
        return this._lost;
    }

    /**
     * Create a WebGPUContext. Throws if WebGPU is not available.
     */
    static async create(): Promise<WebGPUContext> {
        if (!navigator.gpu) {
            throw new Error('WebGPU is not supported in this browser.');
        }

        const adapter = await navigator.gpu.requestAdapter({
            powerPreference: 'high-performance',
        });
        if (!adapter) {
            throw new Error('Failed to obtain a WebGPU adapter. Your GPU may not be supported.');
        }

        const device = await adapter.requestDevice({
            requiredLimits: {
                maxBufferSize: adapter.limits.maxBufferSize,
                maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
            },
        });
        const preferredFormat = navigator.gpu.getPreferredCanvasFormat();

        return new WebGPUContext(adapter, device, preferredFormat);
    }

    /**
     * Configure a canvas element for WebGPU rendering.
     * Returns the GPUCanvasContext for issuing render passes.
     */
    configureCanvas(canvas: HTMLCanvasElement): GPUCanvasContext {
        const ctx = canvas.getContext('webgpu');
        if (!ctx) {
            throw new Error('Failed to get WebGPU context from canvas.');
        }

        ctx.configure({
            device: this.device,
            format: this.preferredFormat,
            alphaMode: 'premultiplied',
        });

        return ctx;
    }

    /** Release GPU resources */
    destroy(): void {
        this.device.destroy();
        this._lost = true;
    }
}
