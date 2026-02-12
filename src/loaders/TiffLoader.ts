import * as UTIF from 'utif2';
import type { IFD } from 'utif2';
import { VolumeData } from '../data/VolumeData.js';

/** Read a numeric tag value from an IFD, returning a fallback if absent */
function tagVal(ifd: IFD, key: string, fallback: number): number {
    const v = ifd[key];
    if (v == null) return fallback;
    if (typeof v === 'number') return v;
    if (Array.isArray(v)) return Number(v[0]) || fallback;
    return fallback;
}

/** Load a TIFF file into a VolumeData */
export async function loadTiff(file: File): Promise<VolumeData> {
    const buffer = await file.arrayBuffer();
    const ifds = UTIF.decode(buffer);

    if (ifds.length === 0) throw new Error('No images found in TIFF file');

    // Multi-page TIFF → stack as 3D volume
    if (ifds.length > 1) {
        return loadMultiPageTiff(buffer, ifds);
    }

    // Single page
    return loadSinglePage(buffer, ifds[0]);
}

/** Stack multiple TIFF pages as z-slices in a 3D volume */
function loadMultiPageTiff(buffer: ArrayBuffer, ifds: IFD[]): VolumeData {
    // Decode first page to determine format
    UTIF.decodeImage(buffer, ifds[0]);
    const width = ifds[0].width;
    const height = ifds[0].height;
    const bps = tagVal(ifds[0], 't258', 8);
    const spp = tagVal(ifds[0], 't277', 1);
    const photo = tagVal(ifds[0], 't262', 1);
    const isGray16 = spp === 1 && (photo === 0 || photo === 1) && bps === 16;
    const numSlices = ifds.length;
    const sliceSize = width * height;

    if (isGray16) {
        const volume = new Uint16Array(sliceSize * numSlices);
        volume.set(new Uint16Array(ifds[0].data.buffer, ifds[0].data.byteOffset, sliceSize), 0);

        for (let z = 1; z < numSlices; z++) {
            UTIF.decodeImage(buffer, ifds[z]);
            if (ifds[z].width !== width || ifds[z].height !== height) {
                throw new Error(`TIFF page ${z} has different dimensions`);
            }
            volume.set(new Uint16Array(ifds[z].data.buffer, ifds[z].data.byteOffset, sliceSize), z * sliceSize);
        }

        return new VolumeData(volume, {
            dimensions: [width, height, numSlices],
            dataType: 'uint16',
            spacing: [1.0, 1.0, 1.0],
        });
    }

    // 8-bit grayscale stack (convert via RGBA)
    const volume = new Uint8Array(sliceSize * numSlices);
    const rgba0 = UTIF.toRGBA8(ifds[0]);
    for (let i = 0, j = 0; i < rgba0.length; i += 4, j++) {
        volume[j] = rgba0[i];
    }

    for (let z = 1; z < numSlices; z++) {
        UTIF.decodeImage(buffer, ifds[z]);
        if (ifds[z].width !== width || ifds[z].height !== height) {
            throw new Error(`TIFF page ${z} has different dimensions`);
        }
        const rgba = UTIF.toRGBA8(ifds[z]);
        const offset = z * sliceSize;
        for (let i = 0, j = 0; i < rgba.length; i += 4, j++) {
            volume[offset + j] = rgba[i];
        }
    }

    return new VolumeData(volume, {
        dimensions: [width, height, numSlices],
        dataType: 'uint8',
        spacing: [1.0, 1.0, 1.0],
    });
}

/** Load a single TIFF page */
function loadSinglePage(buffer: ArrayBuffer, ifd: IFD): VolumeData {
    UTIF.decodeImage(buffer, ifd);

    const width = ifd.width;
    const height = ifd.height;
    const bps = tagVal(ifd, 't258', 8);
    const spp = tagVal(ifd, 't277', 1);
    const photo = tagVal(ifd, 't262', 1);
    const isGrayscale = spp === 1 && (photo === 0 || photo === 1);

    if (isGrayscale && bps === 16) {
        const data = new Uint16Array(ifd.data.buffer, ifd.data.byteOffset, width * height);
        return new VolumeData(data, {
            dimensions: [width, height, 1],
            dataType: 'uint16',
            spacing: [1.0, 1.0, 1.0],
        });
    }

    const rgba = UTIF.toRGBA8(ifd);

    if (isGrayscale) {
        const data = new Uint8Array(width * height);
        for (let i = 0, j = 0; i < rgba.length; i += 4, j++) {
            data[j] = rgba[i];
        }
        return new VolumeData(data, {
            dimensions: [width, height, 1],
            dataType: 'uint8',
            spacing: [1.0, 1.0, 1.0],
        });
    }

    // RGB: store as 3 z-slices
    const sliceSize = width * height;
    const data = new Uint8Array(sliceSize * 3);
    for (let i = 0, j = 0; i < rgba.length; i += 4, j++) {
        data[j] = rgba[i];
        data[j + sliceSize] = rgba[i + 1];
        data[j + sliceSize * 2] = rgba[i + 2];
    }
    return new VolumeData(data, {
        dimensions: [width, height, 3],
        dataType: 'uint8',
        spacing: [1.0, 1.0, 1.0],
        isRGB: true,
    });
}
