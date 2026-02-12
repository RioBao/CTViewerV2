import type { VolumeMetadata, VoxelDataType, VoxelTypedArray } from '../types.js';
import { VolumeData } from '../data/VolumeData.js';
import { StreamingVolumeData } from '../data/StreamingVolumeData.js';
import type { LoadProgress } from './VolumeLoader.js';

const CHUNK_THRESHOLD = 500 * 1024 * 1024; // 500MB
const CHUNK_SIZE = 256 * 1024 * 1024; // 256MB
const PREVIEW_STEP = 4; // Sample every Nth z-slice for preview
const DOWNSAMPLE_SCALE = 4; // 4x downsample for streaming preview

/** Load a RAW binary volume paired with a metadata file */
export async function loadRaw(rawFile: File, metadataFile: File, onProgress?: LoadProgress): Promise<VolumeData> {
    onProgress?.('Reading metadata...');
    const metadata = await parseMetadataFile(metadataFile);

    onProgress?.('Reading volume data...', 0);
    const buffer = await loadBinary(rawFile, (pct) => {
        onProgress?.('Reading volume data...', pct);
    });

    onProgress?.('Processing volume...');
    return VolumeData.fromArrayBuffer(buffer, metadata);
}

/** Load a downsampled preview (every Nth z-slice) for fast display of large RAW files */
export async function loadRawPreview(rawFile: File, metadataFile: File): Promise<VolumeData> {
    const metadata = await parseMetadataFile(metadataFile);
    const [nx, ny, nz] = metadata.dimensions;
    const bytesPerVoxel = metadata.dataType === 'uint8' ? 1 : metadata.dataType === 'uint16' ? 2 : 4;
    const sliceBytes = nx * ny * bytesPerVoxel;

    const step = PREVIEW_STEP;
    const previewNz = Math.ceil(nz / step);
    const previewBuffer = new ArrayBuffer(sliceBytes * previewNz);
    const previewView = new Uint8Array(previewBuffer);

    for (let z = 0, pi = 0; z < nz; z += step, pi++) {
        const offset = z * sliceBytes;
        const chunk = await rawFile.slice(offset, offset + sliceBytes).arrayBuffer();
        previewView.set(new Uint8Array(chunk), pi * sliceBytes);
        // Yield to UI
        if (pi % 4 === 0) await new Promise(r => setTimeout(r, 0));
    }

    const previewMeta: VolumeMetadata = {
        ...metadata,
        dimensions: [nx, ny, previewNz],
        spacing: [metadata.spacing[0], metadata.spacing[1], metadata.spacing[2] * step],
    };

    return VolumeData.fromArrayBuffer(previewBuffer, previewMeta);
}

/**
 * Load a RAW file in streaming mode: parse metadata, build a 4x-downsampled
 * preview by streaming through the file (reading every 4th z-slice, sampling
 * every 4th pixel), and return a StreamingVolumeData that reads slices on demand.
 */
export async function loadRawStreaming(
    rawFile: File,
    metadataFile: File,
    onProgress?: LoadProgress,
): Promise<StreamingVolumeData> {
    onProgress?.('Reading metadata...');
    const metadata = await parseMetadataFile(metadataFile);

    onProgress?.('Building preview...', 0);
    const lowRes = await downsampleVolumeStreaming(rawFile, metadata, DOWNSAMPLE_SCALE, onProgress);

    onProgress?.('Ready (streaming mode)');
    return new StreamingVolumeData(rawFile, metadata, lowRes);
}

/**
 * Stream through a RAW file to build a downsampled VolumeData + compute global min/max
 * without loading the full file into memory.
 * Reads every `scale`th z-slice, samples every `scale`th pixel in x and y.
 */
async function downsampleVolumeStreaming(
    file: File,
    metadata: VolumeMetadata,
    scale: number,
    onProgress?: LoadProgress,
): Promise<VolumeData> {
    const [nx, ny, nz] = metadata.dimensions;
    const bpe = bytesForType(metadata.dataType);
    const sliceBytes = nx * ny * bpe;

    const dnx = Math.ceil(nx / scale);
    const dny = Math.ceil(ny / scale);
    const dnz = Math.ceil(nz / scale);

    const downsampled = new Float32Array(dnx * dny * dnz);
    let globalMin = Infinity;
    let globalMax = -Infinity;

    let di = 0;
    for (let z = 0; z < nz; z += scale) {
        const offset = z * sliceBytes;
        const buf = await file.slice(offset, offset + sliceBytes).arrayBuffer();
        const sliceData = bufferToTypedArray(buf, metadata.dataType, nx * ny);

        for (let y = 0; y < ny; y += scale) {
            for (let x = 0; x < nx; x += scale) {
                const val = sliceData[y * nx + x];
                downsampled[di++] = val;
                if (val < globalMin) globalMin = val;
                if (val > globalMax) globalMax = val;
            }
        }

        // Report progress
        const pct = Math.round(((z / scale + 1) / dnz) * 100);
        onProgress?.('Building preview...', pct);

        // Yield to UI every few slices
        if ((z / scale) % 4 === 0) await new Promise(r => setTimeout(r, 0));
    }

    const downMeta: VolumeMetadata = {
        ...metadata,
        dimensions: [dnx, dny, dnz],
        spacing: [
            metadata.spacing[0] * scale,
            metadata.spacing[1] * scale,
            metadata.spacing[2] * scale,
        ],
        min: globalMin,
        max: globalMax,
    };

    return new VolumeData(downsampled as unknown as VoxelTypedArray, downMeta);
}

/** Bytes per element for a given voxel data type */
function bytesForType(dt: VoxelDataType): number {
    switch (dt) {
        case 'uint8': return 1;
        case 'uint16': return 2;
        case 'float32': return 4;
    }
}

/** Create a typed array view over an ArrayBuffer for the given data type */
function bufferToTypedArray(buf: ArrayBuffer, dt: VoxelDataType, count: number): VoxelTypedArray {
    switch (dt) {
        case 'uint8': return new Uint8Array(buf, 0, count);
        case 'uint16': return new Uint16Array(buf, 0, count);
        case 'float32': return new Float32Array(buf, 0, count);
    }
}

/** Export parseMetadataFile so VolumeLoader can use it */
export { parseMetadataFile };

// ---- Metadata parsers ----

async function parseMetadataFile(file: File): Promise<VolumeMetadata> {
    const ext = file.name.toLowerCase().split('.').pop() ?? '';
    const text = await file.text();

    let metadata: VolumeMetadata;
    if (ext === 'json') {
        metadata = parseJSON(text);
    } else if (ext === 'volumeinfo') {
        metadata = parseVolumeinfo(text);
    } else if (ext === 'dat') {
        metadata = parseDat(text);
    } else {
        throw new Error(`Unknown metadata format: .${ext}`);
    }

    validateMetadata(metadata);
    return metadata;
}

function parseJSON(text: string): VolumeMetadata {
    const obj = JSON.parse(text);
    // Normalize dataType alias
    if (obj.dataType?.toLowerCase() === 'float') obj.dataType = 'float32';
    return {
        dimensions: obj.dimensions,
        dataType: obj.dataType,
        spacing: obj.spacing ?? [1.0, 1.0, 1.0],
        byteOrder: obj.byteOrder,
        isRGB: obj.isRGB,
        min: obj.min,
        max: obj.max,
        description: obj.description,
    };
}

function parseVolumeinfo(text: string): VolumeMetadata {
    const lines = text.split('\n');
    const section: Record<string, string> = {};
    let inVolume = false;

    for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed === '[Volume]') { inVolume = true; continue; }
        if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
            if (inVolume) break;
            continue;
        }
        if (inVolume && trimmed.includes('=')) {
            const eqIdx = trimmed.indexOf('=');
            section[trimmed.slice(0, eqIdx).trim()] = trimmed.slice(eqIdx + 1).trim();
        }
    }

    const sizeX = parseInt(section.SizeX);
    const sizeY = parseInt(section.SizeY);
    const sizeZ = parseInt(section.SizeZ);
    if (isNaN(sizeX) || isNaN(sizeY) || isNaN(sizeZ)) {
        throw new Error('Missing or invalid SizeX/SizeY/SizeZ in volumeinfo');
    }

    let dataType = (section.Datatype as VoxelDataType) || 'uint16';
    // Normalize "float" to "float32"
    if (dataType.toLowerCase() === 'float') dataType = 'float32';

    return {
        dimensions: [sizeX, sizeY, sizeZ],
        dataType,
        spacing: [
            parseFloat(section.VoxelSizeX) || 1.0,
            parseFloat(section.VoxelSizeY) || 1.0,
            parseFloat(section.VoxelSizeZ) || 1.0,
        ],
        byteOrder: 'little-endian',
        min: section.Min !== undefined ? parseFloat(section.Min) : undefined,
        max: section.Max !== undefined ? parseFloat(section.Max) : undefined,
        description: section.Description,
    };
}

function parseDat(text: string): VolumeMetadata {
    const map: Record<string, string> = {};
    for (const line of text.split('\n')) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('#')) continue;
        const idx = trimmed.indexOf(':');
        if (idx === -1) continue;
        map[trimmed.slice(0, idx).trim().toLowerCase()] = trimmed.slice(idx + 1).trim();
    }

    const resolution = (map.resolution || '').split(/\s+/).map(v => parseInt(v, 10)).filter(Number.isFinite);
    if (resolution.length !== 3) {
        throw new Error('Missing or invalid Resolution in DAT file');
    }

    const spacingVals = (map.slicethickness || map.spacing || '').split(/\s+/).map(parseFloat).filter(Number.isFinite);

    const format = (map.format || '').trim().toUpperCase();
    let dataType: VoxelDataType;
    switch (format) {
        case 'UCHAR': case 'UINT8': case 'BYTE': dataType = 'uint8'; break;
        case 'USHORT': case 'UINT16': dataType = 'uint16'; break;
        case 'FLOAT': case 'FLOAT32': dataType = 'float32'; break;
        default: throw new Error(`Unsupported DAT format: ${format || 'unknown'}`);
    }

    return {
        dimensions: resolution as [number, number, number],
        dataType,
        spacing: [spacingVals[0] || 1.0, spacingVals[1] || 1.0, spacingVals[2] || 1.0],
    };
}

function validateMetadata(m: VolumeMetadata): void {
    if (!m.dimensions || m.dimensions.length !== 3) {
        throw new Error('Metadata must contain "dimensions" array with 3 elements');
    }
    if (m.dimensions.some(d => d <= 0)) {
        throw new Error('All dimensions must be positive values');
    }
    if (!m.dataType) {
        throw new Error('Metadata must contain "dataType" field');
    }
    const valid: VoxelDataType[] = ['uint8', 'uint16', 'float32'];
    if (!valid.includes(m.dataType)) {
        throw new Error(`Invalid dataType "${m.dataType}". Must be one of: ${valid.join(', ')}`);
    }
}

// ---- Binary loading ----

async function loadBinary(file: File, onProgress?: (pct: number) => void): Promise<ArrayBuffer> {
    if (file.size > CHUNK_THRESHOLD) {
        return loadChunked(file, onProgress);
    }
    onProgress?.(100);
    return file.arrayBuffer();
}

async function loadChunked(file: File, onProgress?: (pct: number) => void): Promise<ArrayBuffer> {
    const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
    const fullBuffer = new ArrayBuffer(file.size);
    const fullView = new Uint8Array(fullBuffer);

    for (let i = 0; i < totalChunks; i++) {
        const start = i * CHUNK_SIZE;
        const end = Math.min(start + CHUNK_SIZE, file.size);
        const chunkBuffer = await file.slice(start, end).arrayBuffer();
        fullView.set(new Uint8Array(chunkBuffer), start);
        onProgress?.(Math.round(((i + 1) / totalChunks) * 100));
        // Yield to UI between chunks
        await new Promise(r => setTimeout(r, 0));
    }

    return fullBuffer;
}
