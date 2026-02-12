import type { VolumeMetadata } from '../types.js';
import { VolumeData } from '../data/VolumeData.js';

/** Load a NIfTI (.nii or .nii.gz) file into a VolumeData */
export async function loadNifti(file: File): Promise<VolumeData> {
    const isGz = file.name.toLowerCase().endsWith('.nii.gz');

    let buffer = await file.arrayBuffer();
    if (isGz) {
        buffer = await decompressGzip(buffer);
    }

    const { data, metadata } = parseNifti(buffer);
    return new VolumeData(data, metadata);
}

async function decompressGzip(buffer: ArrayBuffer): Promise<ArrayBuffer> {
    if (typeof DecompressionStream === 'undefined') {
        throw new Error('Gzip decompression not supported in this browser');
    }
    const ds = new DecompressionStream('gzip');
    const stream = new Blob([buffer]).stream().pipeThrough(ds);
    return new Response(stream).arrayBuffer();
}

function parseNifti(buffer: ArrayBuffer): { data: Float32Array; metadata: VolumeMetadata } {
    const view = new DataView(buffer);
    let littleEndian = true;

    let headerSize = view.getInt32(0, true);
    if (headerSize !== 348) {
        headerSize = view.getInt32(0, false);
        if (headerSize !== 348) throw new Error('Invalid NIfTI header');
        littleEndian = false;
    }

    const dim0 = view.getInt16(40, littleEndian);
    const dim1 = view.getInt16(42, littleEndian);
    const dim2 = view.getInt16(44, littleEndian);
    const dim3 = view.getInt16(46, littleEndian);
    const dim4 = view.getInt16(48, littleEndian);

    if (dim0 < 3 || dim1 <= 0 || dim2 <= 0 || dim3 <= 0) {
        throw new Error('Invalid NIfTI dimensions');
    }

    const datatype = view.getInt16(70, littleEndian);
    const voxOffset = view.getFloat32(108, littleEndian);
    const slope = view.getFloat32(112, littleEndian);
    const intercept = view.getFloat32(116, littleEndian);

    const pixdim: number[] = [];
    for (let i = 0; i < 8; i++) {
        pixdim.push(view.getFloat32(76 + i * 4, littleEndian));
    }

    const spacing: [number, number, number] = [
        pixdim[1] > 0 ? pixdim[1] : 1.0,
        pixdim[2] > 0 ? pixdim[2] : 1.0,
        pixdim[3] > 0 ? pixdim[3] : 1.0,
    ];

    const bpv = bytesPerVoxel(datatype);
    if (!bpv) throw new Error(`Unsupported NIfTI datatype: ${datatype}`);

    const totalVoxels = dim1 * dim2 * dim3;
    const dataOffset = Math.max(0, Math.floor(voxOffset));
    const dataView = new DataView(buffer, dataOffset);

    const s = (slope === 0 || !Number.isFinite(slope)) ? 1 : slope;
    const inter = Number.isFinite(intercept) ? intercept : 0;

    const out = new Float32Array(totalVoxels);
    let bytePos = 0;

    if (datatype === 128) {
        // RGB24 → grayscale via BT.601
        for (let idx = 0; idx < totalVoxels; idx++) {
            const r = dataView.getUint8(bytePos);
            const g = dataView.getUint8(bytePos + 1);
            const b = dataView.getUint8(bytePos + 2);
            out[idx] = (0.299 * r + 0.587 * g + 0.114 * b) * s + inter;
            bytePos += 3;
        }
    } else {
        for (let idx = 0; idx < totalVoxels; idx++) {
            out[idx] = readValue(dataView, bytePos, datatype, littleEndian) * s + inter;
            bytePos += bpv;
        }
    }

    return {
        data: out,
        metadata: {
            dimensions: [dim1, dim2, dim3],
            dataType: 'float32',
            spacing,
            description: dim4 > 1 ? 'NIfTI (first volume)' : 'NIfTI',
        },
    };
}

function bytesPerVoxel(dt: number): number {
    switch (dt) {
        case 2: return 1;    // uint8
        case 4: return 2;    // int16
        case 8: return 4;    // int32
        case 16: return 4;   // float32
        case 64: return 8;   // float64
        case 128: return 3;  // RGB24
        case 256: return 1;  // int8
        case 512: return 2;  // uint16
        case 768: return 4;  // uint32
        default: return 0;
    }
}

function readValue(view: DataView, offset: number, dt: number, le: boolean): number {
    switch (dt) {
        case 2: return view.getUint8(offset);
        case 4: return view.getInt16(offset, le);
        case 8: return view.getInt32(offset, le);
        case 16: return view.getFloat32(offset, le);
        case 64: return view.getFloat64(offset, le);
        case 256: return view.getInt8(offset);
        case 512: return view.getUint16(offset, le);
        case 768: return view.getUint32(offset, le);
        default: return 0;
    }
}
