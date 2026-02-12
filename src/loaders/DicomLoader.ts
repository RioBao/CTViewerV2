import { VolumeData } from '../data/VolumeData.js';
import type { FileGroup } from '../app/FilePicker.js';

const LONG_LENGTH_VR = new Set(['OB', 'OW', 'OF', 'SQ', 'UT', 'UN']);

const TAG_VR: Record<string, string> = {
    '0008,0060': 'CS', '0008,103e': 'LO', '0018,0050': 'DS',
    '0018,0088': 'DS', '0020,000e': 'UI', '0020,0013': 'IS',
    '0020,0032': 'DS', '0020,0037': 'DS', '0028,0002': 'US',
    '0028,0004': 'CS', '0028,0008': 'IS', '0028,0010': 'US',
    '0028,0011': 'US', '0028,0030': 'DS', '0028,0100': 'US',
    '0028,0101': 'US', '0028,0103': 'US', '0028,1052': 'DS',
    '0028,1053': 'DS',
};

interface SliceInfo {
    rows: number;
    cols: number;
    bitsAllocated: number;
    pixelRepresentation: number;
    samplesPerPixel: number;
    pixelSpacing: number[] | null;
    imagePosition: number[] | null;
    imageOrientation: number[] | null;
    instanceNumber: number | null;
    seriesDescription: string | null;
    rescaleIntercept: number;
    rescaleSlope: number;
    sliceThickness: number | null;
    spacingBetweenSlices: number | null;
    numberOfFrames: number;
    pixelData: Int16Array | Uint16Array | Uint8Array | null;
    explicitVR: boolean;
    littleEndian: boolean;
}

interface Element {
    tagKey: string;
    vr: string;
    length: number;
    valueOffset: number;
    nextOffset: number;
}

/** Load a DICOM series (one or more .dcm files) into a VolumeData */
export async function loadDicom(group: FileGroup): Promise<VolumeData> {
    if (group.files.length === 0) throw new Error('No DICOM files provided');

    const sliceInfos: SliceInfo[] = [];
    for (const file of group.files) {
        const info = await parseDicomFile(file);
        if (info) sliceInfos.push(info);
    }

    if (sliceInfos.length === 0) {
        throw new Error('No valid DICOM image files found');
    }

    // Multi-frame single file
    if (sliceInfos.length === 1 && sliceInfos[0].numberOfFrames > 1) {
        return buildMultiFrameVolume(sliceInfos[0]);
    }

    return buildSeriesVolume(sliceInfos);
}

// ---- Volume building ----

function buildSeriesVolume(sliceInfos: SliceInfo[]): VolumeData {
    if (sliceInfos.length === 0) throw new Error('No DICOM slices parsed');

    const ref = sliceInfos[0];
    const { rows, cols } = ref;
    const sliceSize = rows * cols;

    for (const info of sliceInfos) {
        if (info.rows !== rows || info.cols !== cols) {
            throw new Error('DICOM series has inconsistent slice dimensions');
        }
    }

    const sorted = sortSlices(sliceInfos);
    const numSlices = sorted.length;
    const volumeData = new Float32Array(sliceSize * numSlices);
    let min = Infinity;
    let max = -Infinity;

    for (let z = 0; z < numSlices; z++) {
        const slice = sorted[z];
        const scaled = applyRescale(slice.pixelData!, slice.rescaleSlope, slice.rescaleIntercept);
        volumeData.set(scaled, z * sliceSize);
        for (let i = 0; i < scaled.length; i++) {
            if (scaled[i] < min) min = scaled[i];
            if (scaled[i] > max) max = scaled[i];
        }
    }

    const spacing = computeSpacing(sorted, ref);
    return new VolumeData(volumeData, {
        dimensions: [cols, rows, numSlices],
        dataType: 'float32',
        spacing,
        min, max,
        description: ref.seriesDescription || 'DICOM Series',
    });
}

function buildMultiFrameVolume(info: SliceInfo): VolumeData {
    const { rows, cols, numberOfFrames } = info;
    const sliceSize = rows * cols;
    if (!info.pixelData || info.pixelData.length < sliceSize * numberOfFrames) {
        throw new Error('Multi-frame pixel data length mismatch');
    }

    const volumeData = new Float32Array(sliceSize * numberOfFrames);
    let min = Infinity;
    let max = -Infinity;

    for (let f = 0; f < numberOfFrames; f++) {
        const offset = f * sliceSize;
        const frameView = info.pixelData.subarray(offset, offset + sliceSize);
        const scaled = applyRescale(frameView, info.rescaleSlope, info.rescaleIntercept);
        volumeData.set(scaled, offset);
        for (let i = 0; i < scaled.length; i++) {
            if (scaled[i] < min) min = scaled[i];
            if (scaled[i] > max) max = scaled[i];
        }
    }

    const spacing = computeSpacing([info], info);
    return new VolumeData(volumeData, {
        dimensions: [cols, rows, numberOfFrames],
        dataType: 'float32',
        spacing,
        min, max,
        description: info.seriesDescription || 'DICOM Multi-frame',
    });
}

function applyRescale(
    data: Int16Array | Uint16Array | Uint8Array,
    slope: number,
    intercept: number,
): Float32Array {
    const s = (!slope || slope === 0) ? 1 : slope;
    const out = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) {
        out[i] = data[i] * s + intercept;
    }
    return out;
}

// ---- Slice sorting ----

function sortSlices(sliceInfos: SliceInfo[]): SliceInfo[] {
    const withPos = sliceInfos.filter(s => s.imagePosition?.length === 3);
    const hasOrient = sliceInfos[0].imageOrientation?.length === 6;

    if (withPos.length === sliceInfos.length && hasOrient) {
        const normal = computeNormal(sliceInfos[0].imageOrientation!);
        return sliceInfos
            .map(s => ({ info: s, loc: dot(s.imagePosition!, normal) }))
            .sort((a, b) => a.loc - b.loc)
            .map(s => s.info);
    }

    if (withPos.length === sliceInfos.length) {
        return sliceInfos
            .map(s => ({ info: s, loc: s.imagePosition![2] }))
            .sort((a, b) => a.loc - b.loc)
            .map(s => s.info);
    }

    if (sliceInfos.every(s => s.instanceNumber != null)) {
        return sliceInfos.slice().sort((a, b) => a.instanceNumber! - b.instanceNumber!);
    }

    return sliceInfos;
}

function computeSpacing(sorted: SliceInfo[], ref: SliceInfo): [number, number, number] {
    let spacingZ = 1.0;
    const positions = sorted
        .filter(s => s.imagePosition?.length === 3)
        .map(s => s.imagePosition!);

    if (positions.length >= 2) {
        if (ref.imageOrientation?.length === 6) {
            const normal = computeNormal(ref.imageOrientation);
            const locs = positions.map(p => dot(p, normal));
            spacingZ = averageDiffs(locs);
        } else {
            const dists: number[] = [];
            for (let i = 1; i < positions.length; i++) {
                const dx = positions[i][0] - positions[i - 1][0];
                const dy = positions[i][1] - positions[i - 1][1];
                const dz = positions[i][2] - positions[i - 1][2];
                dists.push(Math.sqrt(dx * dx + dy * dy + dz * dz));
            }
            spacingZ = average(dists) || 1.0;
        }
    } else if (ref.spacingBetweenSlices) {
        spacingZ = ref.spacingBetweenSlices;
    } else if (ref.sliceThickness) {
        spacingZ = ref.sliceThickness;
    }

    const ps = ref.pixelSpacing || [1.0, 1.0];
    return [ps[1] || 1.0, ps[0] || 1.0, spacingZ || 1.0];
}

function computeNormal(orient: number[]): number[] {
    const r = orient.slice(0, 3);
    const c = orient.slice(3, 6);
    return [
        r[1] * c[2] - r[2] * c[1],
        r[2] * c[0] - r[0] * c[2],
        r[0] * c[1] - r[1] * c[0],
    ];
}

function dot(a: number[], b: number[]): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function averageDiffs(values: number[]): number {
    if (values.length < 2) return 1.0;
    const diffs: number[] = [];
    for (let i = 1; i < values.length; i++) diffs.push(Math.abs(values[i] - values[i - 1]));
    return average(diffs) || 1.0;
}

function average(values: number[]): number {
    if (values.length === 0) return 0;
    return values.reduce((a, b) => a + b, 0) / values.length;
}

// ---- DICOM binary parser ----

async function parseDicomFile(file: File): Promise<SliceInfo | null> {
    const buffer = await file.arrayBuffer();
    const view = new DataView(buffer);

    let offset = 0;
    const meta: { transferSyntaxUID?: string } = {};
    let explicitVR = true;
    let littleEndian = true;

    if (hasPreamble(view)) {
        offset = 132;
        offset = parseMetaHeader(view, offset, meta);
        const tsUID = meta.transferSyntaxUID || '1.2.840.10008.1.2.1';
        const ts = getTransferSyntax(tsUID);
        if (!ts) {
            console.warn(`Skipping ${file.name}: unsupported transfer syntax ${tsUID}`);
            return null;
        }
        if (ts.compressed) {
            console.warn(`Skipping ${file.name}: compressed DICOM not supported`);
            return null;
        }
        if (!ts.littleEndian) {
            console.warn(`Skipping ${file.name}: big-endian DICOM not supported`);
            return null;
        }
        explicitVR = ts.explicitVR;
        littleEndian = ts.littleEndian;
    } else {
        explicitVR = guessExplicitVR(view);
    }

    const info: SliceInfo = {
        rows: 0, cols: 0, bitsAllocated: 0,
        pixelRepresentation: 0, samplesPerPixel: 1,
        pixelSpacing: null, imagePosition: null, imageOrientation: null,
        instanceNumber: null, seriesDescription: null,
        rescaleIntercept: 0, rescaleSlope: 1,
        sliceThickness: null, spacingBetweenSlices: null,
        numberOfFrames: 1, pixelData: null,
        explicitVR, littleEndian,
    };

    let pixelDataOffset: number | null = null;
    let pixelDataLength: number | null = null;

    while (offset < view.byteLength) {
        const el = readElement(view, offset, info.explicitVR, info.littleEndian);
        if (!el) break;

        if (el.tagKey === '7fe0,0010') {
            if (el.length === 0xffffffff) {
                console.warn(`Skipping ${file.name}: encapsulated pixel data not supported`);
                return null;
            }
            pixelDataOffset = el.valueOffset;
            pixelDataLength = el.length;
            break;
        }

        if (el.length === 0xffffffff) {
            offset = skipUndefinedLength(view, el.valueOffset, info.explicitVR, info.littleEndian);
            continue;
        }

        readTagValue(view, el, info);
        offset = el.nextOffset;
    }

    // Skip non-image DICOM files (e.g. RT Structure Sets, RT Plans)
    if (!info.rows || !info.cols || !info.bitsAllocated) {
        console.warn(`Skipping ${file.name}: not an image file (no rows/cols/bitsAllocated)`);
        return null;
    }
    if (info.samplesPerPixel !== 1) {
        console.warn(`Skipping ${file.name}: non-monochrome (samplesPerPixel=${info.samplesPerPixel})`);
        return null;
    }

    info.pixelData = extractPixelData(view, info.bitsAllocated, info.pixelRepresentation, pixelDataOffset, pixelDataLength);
    return info;
}

function extractPixelData(
    view: DataView, bitsAllocated: number, pixelRep: number,
    dataOffset: number | null, dataLength: number | null,
): Int16Array | Uint16Array | Uint8Array {
    if (dataOffset === null || dataLength === null) throw new Error('Pixel data not found');
    if (dataOffset + dataLength > view.byteLength) throw new Error('Pixel data extends beyond file');
    if (bitsAllocated !== 8 && bitsAllocated !== 16) throw new Error(`Unsupported BitsAllocated: ${bitsAllocated}`);

    const buf = view.buffer.slice(dataOffset, dataOffset + dataLength);
    if (bitsAllocated === 8) return new Uint8Array(buf);
    return pixelRep === 1 ? new Int16Array(buf) : new Uint16Array(buf);
}

// ---- Low-level element reading ----

function hasPreamble(view: DataView): boolean {
    return view.byteLength >= 132 && view.getUint32(128, false) === 0x4449434D;
}

function guessExplicitVR(view: DataView): boolean {
    if (view.byteLength < 8) return true;
    const vr = String.fromCharCode(view.getUint8(4), view.getUint8(5));
    const valid = new Set([
        'AE','AS','AT','CS','DA','DS','DT','FD','FL','IS','LO','LT',
        'OB','OD','OF','OL','OW','PN','SH','SL','SQ','SS','ST','TM',
        'UC','UI','UL','UN','UR','US','UT',
    ]);
    return valid.has(vr);
}

function getTransferSyntax(uid: string): { explicitVR: boolean; littleEndian: boolean; compressed: boolean } | null {
    const map: Record<string, { explicitVR: boolean; littleEndian: boolean; compressed: boolean }> = {
        '1.2.840.10008.1.2': { explicitVR: false, littleEndian: true, compressed: false },
        '1.2.840.10008.1.2.1': { explicitVR: true, littleEndian: true, compressed: false },
        '1.2.840.10008.1.2.2': { explicitVR: true, littleEndian: false, compressed: false },
    };
    return map[uid] ?? null;
}

function parseMetaHeader(view: DataView, startOffset: number, meta: { transferSyntaxUID?: string }): number {
    let offset = startOffset;
    while (offset + 8 < view.byteLength) {
        const group = view.getUint16(offset, true);
        if (group !== 0x0002) break;
        const el = readElement(view, offset, true, true);
        if (!el) break;
        if (el.tagKey === '0002,0010') {
            meta.transferSyntaxUID = readString(view, el.valueOffset, el.length);
        }
        offset = el.nextOffset;
    }
    return offset;
}

function readElement(view: DataView, offset: number, explicitVR: boolean, littleEndian: boolean): Element | null {
    if (offset + 8 > view.byteLength) return null;

    const group = view.getUint16(offset, littleEndian);
    const element = view.getUint16(offset + 2, littleEndian);
    const tagKey = tagKeyStr(group, element);
    offset += 4;

    let vr: string;
    let length: number;

    if (explicitVR) {
        vr = String.fromCharCode(view.getUint8(offset), view.getUint8(offset + 1));
        offset += 2;
        if (LONG_LENGTH_VR.has(vr)) {
            offset += 2; // reserved
            length = view.getUint32(offset, littleEndian);
            offset += 4;
        } else {
            length = view.getUint16(offset, littleEndian);
            offset += 2;
        }
    } else {
        vr = TAG_VR[tagKey] || 'UN';
        length = view.getUint32(offset, littleEndian);
        offset += 4;
    }

    return {
        tagKey, vr, length,
        valueOffset: offset,
        nextOffset: length === 0xffffffff ? offset : offset + length,
    };
}

function skipUndefinedLength(view: DataView, offset: number, explicitVR: boolean, littleEndian: boolean): number {
    let cursor = offset;
    while (cursor + 8 <= view.byteLength) {
        const g = view.getUint16(cursor, littleEndian);
        const e = view.getUint16(cursor + 2, littleEndian);
        const len = view.getUint32(cursor + 4, littleEndian);
        cursor += 8;

        if (g === 0xfffe && e === 0xe0dd) return cursor + len;
        if (g === 0xfffe && e === 0xe000) {
            if (len === 0xffffffff) {
                cursor = skipItemUndefinedLength(view, cursor, explicitVR, littleEndian);
            } else {
                cursor += len;
            }
            continue;
        }
        cursor += len;
    }
    return cursor;
}

function skipItemUndefinedLength(view: DataView, offset: number, explicitVR: boolean, littleEndian: boolean): number {
    let cursor = offset;
    while (cursor + 8 <= view.byteLength) {
        const g = view.getUint16(cursor, littleEndian);
        const e = view.getUint16(cursor + 2, littleEndian);
        if (g === 0xfffe && e === 0xe00d) return cursor + 8;

        const el = readElement(view, cursor, explicitVR, littleEndian);
        if (!el) return cursor;
        cursor = el.length === 0xffffffff
            ? skipUndefinedLength(view, el.valueOffset, explicitVR, littleEndian)
            : el.nextOffset;
    }
    return cursor;
}

function readTagValue(view: DataView, el: Element, info: SliceInfo): void {
    switch (el.tagKey) {
        case '0008,103e': info.seriesDescription = readString(view, el.valueOffset, el.length); break;
        case '0018,0050': info.sliceThickness = readNumber(view, el.valueOffset, el.length); break;
        case '0018,0088': info.spacingBetweenSlices = readNumber(view, el.valueOffset, el.length); break;
        case '0020,0013': info.instanceNumber = readIntString(view, el.valueOffset, el.length); break;
        case '0020,0032': info.imagePosition = readNumberList(view, el.valueOffset, el.length, 3); break;
        case '0020,0037': info.imageOrientation = readNumberList(view, el.valueOffset, el.length, 6); break;
        case '0028,0002': info.samplesPerPixel = view.getUint16(el.valueOffset, info.littleEndian); break;
        case '0028,0008': info.numberOfFrames = readIntString(view, el.valueOffset, el.length) || 1; break;
        case '0028,0010': info.rows = view.getUint16(el.valueOffset, info.littleEndian); break;
        case '0028,0011': info.cols = view.getUint16(el.valueOffset, info.littleEndian); break;
        case '0028,0030': info.pixelSpacing = readNumberList(view, el.valueOffset, el.length, 2); break;
        case '0028,0100': info.bitsAllocated = view.getUint16(el.valueOffset, info.littleEndian); break;
        case '0028,0103': info.pixelRepresentation = view.getUint16(el.valueOffset, info.littleEndian); break;
        case '0028,1052': info.rescaleIntercept = readNumber(view, el.valueOffset, el.length) ?? 0; break;
        case '0028,1053': info.rescaleSlope = readNumber(view, el.valueOffset, el.length) ?? 1; break;
    }
}

// ---- String/number readers ----

function readString(view: DataView, offset: number, length: number): string {
    const bytes = new Uint8Array(view.buffer, offset, length);
    let result = '';
    for (let i = 0; i < bytes.length; i++) {
        if (bytes[i] === 0) break;
        result += String.fromCharCode(bytes[i]);
    }
    return result.trim();
}

function readNumber(view: DataView, offset: number, length: number): number | null {
    const val = parseFloat(readString(view, offset, length));
    return Number.isFinite(val) ? val : null;
}

function readIntString(view: DataView, offset: number, length: number): number | null {
    const val = parseInt(readString(view, offset, length), 10);
    return Number.isFinite(val) ? val : null;
}

function readNumberList(view: DataView, offset: number, length: number, expected: number): number[] | null {
    const parts = readString(view, offset, length).split('\\').map(parseFloat).filter(Number.isFinite);
    return parts.length >= expected ? parts.slice(0, expected) : (parts.length ? parts : null);
}

function tagKeyStr(group: number, element: number): string {
    return `${group.toString(16).padStart(4, '0')},${element.toString(16).padStart(4, '0')}`;
}
