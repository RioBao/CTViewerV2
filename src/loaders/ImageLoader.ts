import { VolumeData } from '../data/VolumeData.js';

/** Load a 2D image (PNG, JPG, etc.) into a VolumeData */
export async function loadImage(file: File): Promise<VolumeData> {
    const url = URL.createObjectURL(file);

    try {
        const img = await decodeImage(url);
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(img, 0, 0);

        const { data: pixels } = ctx.getImageData(0, 0, img.width, img.height);
        const { width, height } = img;

        // Detect grayscale (R === G === B for all pixels)
        let isGrayscale = true;
        for (let i = 0; i < pixels.length; i += 4) {
            if (pixels[i] !== pixels[i + 1] || pixels[i] !== pixels[i + 2]) {
                isGrayscale = false;
                break;
            }
        }

        if (isGrayscale) {
            const data = new Uint8Array(width * height);
            for (let i = 0, j = 0; i < pixels.length; i += 4, j++) {
                data[j] = pixels[i];
            }
            return new VolumeData(data, {
                dimensions: [width, height, 1],
                dataType: 'uint8',
                spacing: [1.0, 1.0, 1.0],
            });
        }

        // Color: R, G, B as separate z-slices
        const sliceSize = width * height;
        const data = new Uint8Array(sliceSize * 3);
        for (let i = 0, j = 0; i < pixels.length; i += 4, j++) {
            data[j] = pixels[i];                     // R → z=0
            data[j + sliceSize] = pixels[i + 1];     // G → z=1
            data[j + sliceSize * 2] = pixels[i + 2]; // B → z=2
        }
        return new VolumeData(data, {
            dimensions: [width, height, 3],
            dataType: 'uint8',
            spacing: [1.0, 1.0, 1.0],
            isRGB: true,
        });
    } finally {
        URL.revokeObjectURL(url);
    }
}

function decodeImage(url: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error('Failed to load image'));
        img.src = url;
    });
}
