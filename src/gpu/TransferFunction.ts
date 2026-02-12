export interface TFPreset {
    name: string;
    stops: Array<{ position: number; color: [number, number, number, number] }>;
}

export const TF_PRESETS: TFPreset[] = [
    {
        name: 'CT Bone',
        stops: [
            { position: 0.0, color: [0, 0, 0, 0] },
            { position: 0.3, color: [0, 0, 0, 0] },
            { position: 0.45, color: [180, 60, 30, 20] },
            { position: 0.65, color: [255, 200, 150, 120] },
            { position: 0.85, color: [255, 240, 220, 200] },
            { position: 1.0, color: [255, 255, 255, 255] },
        ],
    },
    {
        name: 'CT Soft Tissue',
        stops: [
            { position: 0.0, color: [0, 0, 0, 0] },
            { position: 0.15, color: [0, 0, 0, 0] },
            { position: 0.3, color: [180, 50, 50, 40] },
            { position: 0.5, color: [220, 130, 100, 100] },
            { position: 0.7, color: [255, 200, 170, 60] },
            { position: 1.0, color: [255, 255, 255, 30] },
        ],
    },
    {
        name: 'Hot Metal',
        stops: [
            { position: 0.0, color: [0, 0, 0, 0] },
            { position: 0.2, color: [30, 0, 0, 10] },
            { position: 0.4, color: [180, 30, 0, 60] },
            { position: 0.6, color: [255, 130, 0, 120] },
            { position: 0.8, color: [255, 220, 80, 180] },
            { position: 1.0, color: [255, 255, 200, 255] },
        ],
    },
    {
        name: 'Cool-Warm',
        stops: [
            { position: 0.0, color: [60, 60, 200, 0] },
            { position: 0.2, color: [60, 100, 220, 30] },
            { position: 0.4, color: [100, 180, 220, 80] },
            { position: 0.5, color: [200, 200, 200, 40] },
            { position: 0.6, color: [220, 160, 100, 80] },
            { position: 0.8, color: [220, 80, 60, 30] },
            { position: 1.0, color: [200, 40, 40, 0] },
        ],
    },
    {
        name: 'Grayscale',
        stops: [
            { position: 0.0, color: [0, 0, 0, 0] },
            { position: 0.1, color: [25, 25, 25, 10] },
            { position: 0.5, color: [128, 128, 128, 128] },
            { position: 1.0, color: [255, 255, 255, 255] },
        ],
    },
];

/**
 * Manages a 256×1 RGBA transfer function texture for volume compositing.
 */
export class TransferFunction {
    private device: GPUDevice;
    private _texture: GPUTexture;
    private _sampler: GPUSampler;

    constructor(device: GPUDevice) {
        this.device = device;

        this._texture = device.createTexture({
            size: [256, 1],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });

        this._sampler = device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
        });

        // Initialize with default grayscale ramp
        this.applyPreset(TF_PRESETS[TF_PRESETS.length - 1]);
    }

    get texture(): GPUTexture { return this._texture; }
    get sampler(): GPUSampler { return this._sampler; }

    /** Interpolate preset stops into 256×4 RGBA and upload */
    applyPreset(preset: TFPreset): void {
        const data = new Uint8Array(256 * 4);
        const stops = preset.stops;

        for (let i = 0; i < 256; i++) {
            const t = i / 255;

            // Find bracketing stops
            let lo = 0;
            let hi = stops.length - 1;
            for (let s = 0; s < stops.length - 1; s++) {
                if (t >= stops[s].position && t <= stops[s + 1].position) {
                    lo = s;
                    hi = s + 1;
                    break;
                }
            }

            const range = stops[hi].position - stops[lo].position;
            const f = range > 0 ? (t - stops[lo].position) / range : 0;

            const idx = i * 4;
            data[idx + 0] = Math.round(stops[lo].color[0] + (stops[hi].color[0] - stops[lo].color[0]) * f);
            data[idx + 1] = Math.round(stops[lo].color[1] + (stops[hi].color[1] - stops[lo].color[1]) * f);
            data[idx + 2] = Math.round(stops[lo].color[2] + (stops[hi].color[2] - stops[lo].color[2]) * f);
            data[idx + 3] = Math.round(stops[lo].color[3] + (stops[hi].color[3] - stops[lo].color[3]) * f);
        }

        this.device.queue.writeTexture(
            { texture: this._texture },
            data.buffer,
            { bytesPerRow: 256 * 4 },
            [256, 1],
        );
    }

    destroy(): void {
        this._texture.destroy();
    }
}
