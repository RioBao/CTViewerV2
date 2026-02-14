import type { MaskClassDataType, MaskVolume } from './MaskTypes.js';

export interface BinaryMaskRLE {
    encoding: 'binary-rle-v1';
    totalVoxels: number;
    oneCount: number;
    startsWith: 0 | 1;
    runs: Uint32Array;
}

export interface BinaryMaskRLEJson {
    encoding: 'binary-rle-v1';
    totalVoxels: number;
    oneCount: number;
    startsWith: 0 | 1;
    runs: number[];
}

export interface LabelChunkRLE {
    encoding: 'label-rle-v1';
    totalVoxels: number;
    nonZeroVoxels: number;
    values: Uint16Array;
    lengths: Uint32Array;
}

export interface LabelChunkRLEJson {
    encoding: 'label-rle-v1';
    totalVoxels: number;
    nonZeroVoxels: number;
    values: number[];
    lengths: number[];
}

function coerceBinaryBit(value: number): 0 | 1 {
    return value !== 0 ? 1 : 0;
}

/**
 * Encode voxels for one class as a binary run-length stream in XYZ linear order.
 * This keeps export payloads compact for sparse masks while preserving exact labels.
 */
export function encodeClassMaskRLE(mask: MaskVolume, classId: number): BinaryMaskRLE {
    const [nx, ny, nz] = mask.dimensions;
    const totalVoxels = nx * ny * nz;
    if (totalVoxels <= 0) {
        return {
            encoding: 'binary-rle-v1',
            totalVoxels: 0,
            oneCount: 0,
            startsWith: 0,
            runs: new Uint32Array(0),
        };
    }

    let oneCount = 0;
    const runs: number[] = [];
    let firstBit: 0 | 1 = mask.getVoxel(0, 0, 0) === classId ? 1 : 0;
    let currentBit: 0 | 1 = firstBit;
    let runLength = 0;

    for (let z = 0; z < nz; z++) {
        for (let y = 0; y < ny; y++) {
            for (let x = 0; x < nx; x++) {
                const bit: 0 | 1 = mask.getVoxel(x, y, z) === classId ? 1 : 0;
                if (bit === 1) oneCount++;
                if (bit === currentBit) {
                    runLength++;
                    continue;
                }
                runs.push(runLength);
                currentBit = bit;
                runLength = 1;
            }
        }
    }
    runs.push(runLength);

    return {
        encoding: 'binary-rle-v1',
        totalVoxels,
        oneCount,
        startsWith: firstBit,
        runs: Uint32Array.from(runs),
    };
}

/**
 * Encode a binary mask (0/1 byte buffer) with run-length encoding.
 */
export function encodeBinaryMaskBitsRLE(bits: Uint8Array): BinaryMaskRLE {
    const totalVoxels = bits.length;
    if (totalVoxels <= 0) {
        return {
            encoding: 'binary-rle-v1',
            totalVoxels: 0,
            oneCount: 0,
            startsWith: 0,
            runs: new Uint32Array(0),
        };
    }

    const runs: number[] = [];
    const startsWith = coerceBinaryBit(bits[0]);
    let currentBit = startsWith;
    let runLength = 0;
    let oneCount = 0;

    for (let i = 0; i < bits.length; i++) {
        const bit = coerceBinaryBit(bits[i]);
        if (bit === 1) oneCount++;
        if (bit === currentBit) {
            runLength++;
            continue;
        }
        runs.push(runLength);
        currentBit = bit;
        runLength = 1;
    }
    runs.push(runLength);

    return {
        encoding: 'binary-rle-v1',
        totalVoxels,
        oneCount,
        startsWith,
        runs: Uint32Array.from(runs),
    };
}

/**
 * Decode binary run-length stream back to a 0/1 byte buffer.
 */
export function decodeBinaryMaskBitsRLE(input: {
    totalVoxels: number;
    startsWith: 0 | 1;
    runs: Uint32Array | number[];
}): Uint8Array {
    const total = Math.max(0, Math.floor(input.totalVoxels));
    const out = new Uint8Array(total);
    if (total === 0) return out;

    let write = 0;
    let bit: 0 | 1 = input.startsWith === 1 ? 1 : 0;
    for (let i = 0; i < input.runs.length && write < total; i++) {
        const length = Math.max(0, Math.floor(input.runs[i] ?? 0));
        if (length <= 0) {
            bit = bit === 1 ? 0 : 1;
            continue;
        }
        if (bit === 1) {
            out.fill(1, write, Math.min(total, write + length));
        }
        write += length;
        bit = bit === 1 ? 0 : 1;
    }
    return out;
}

export function binaryMaskRLEToJSON(rle: BinaryMaskRLE): BinaryMaskRLEJson {
    return {
        encoding: rle.encoding,
        totalVoxels: rle.totalVoxels,
        oneCount: rle.oneCount,
        startsWith: rle.startsWith,
        runs: Array.from(rle.runs),
    };
}

export function binaryMaskRLEFromJSON(json: BinaryMaskRLEJson): BinaryMaskRLE {
    return {
        encoding: json.encoding,
        totalVoxels: json.totalVoxels,
        oneCount: json.oneCount,
        startsWith: json.startsWith,
        runs: Uint32Array.from(json.runs),
    };
}

/**
 * Encode class-id labels (uint8/uint16 values) with run-length encoding.
 */
export function encodeLabelValuesRLE(values: Uint8Array | Uint16Array): LabelChunkRLE {
    const totalVoxels = values.length;
    if (totalVoxels <= 0) {
        return {
            encoding: 'label-rle-v1',
            totalVoxels: 0,
            nonZeroVoxels: 0,
            values: new Uint16Array(0),
            lengths: new Uint32Array(0),
        };
    }

    const runValues: number[] = [];
    const runLengths: number[] = [];
    let nonZeroVoxels = 0;
    let current = values[0];
    let length = 1;
    if (current !== 0) nonZeroVoxels++;

    for (let i = 1; i < totalVoxels; i++) {
        const next = values[i];
        if (next !== 0) nonZeroVoxels++;
        if (next === current) {
            length++;
            continue;
        }
        runValues.push(current);
        runLengths.push(length);
        current = next;
        length = 1;
    }
    runValues.push(current);
    runLengths.push(length);

    return {
        encoding: 'label-rle-v1',
        totalVoxels,
        nonZeroVoxels,
        values: Uint16Array.from(runValues),
        lengths: Uint32Array.from(runLengths),
    };
}

/**
 * Decode class-id label RLE stream to a typed class buffer.
 */
export function decodeLabelValuesRLE(
    data: { totalVoxels: number; values: Uint16Array | number[]; lengths: Uint32Array | number[] },
    classDataType: MaskClassDataType,
): Uint8Array | Uint16Array {
    const total = Math.max(0, Math.floor(data.totalVoxels));
    const out = classDataType === 'uint16' ? new Uint16Array(total) : new Uint8Array(total);
    if (total === 0) return out;

    let write = 0;
    const runs = Math.min(data.values.length, data.lengths.length);
    for (let i = 0; i < runs && write < total; i++) {
        const value = Math.max(0, Math.floor(data.values[i] ?? 0));
        const length = Math.max(0, Math.floor(data.lengths[i] ?? 0));
        if (length <= 0) continue;
        out.fill(value, write, Math.min(total, write + length));
        write += length;
    }
    return out;
}

export function labelChunkRLEToJSON(rle: LabelChunkRLE): LabelChunkRLEJson {
    return {
        encoding: rle.encoding,
        totalVoxels: rle.totalVoxels,
        nonZeroVoxels: rle.nonZeroVoxels,
        values: Array.from(rle.values),
        lengths: Array.from(rle.lengths),
    };
}

export function labelChunkRLEFromJSON(json: LabelChunkRLEJson): LabelChunkRLE {
    return {
        encoding: json.encoding,
        totalVoxels: json.totalVoxels,
        nonZeroVoxels: json.nonZeroVoxels,
        values: Uint16Array.from(json.values),
        lengths: Uint32Array.from(json.lengths),
    };
}
