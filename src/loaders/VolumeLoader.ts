import type { FileGroup } from '../app/FilePicker.js';
import { VolumeData } from '../data/VolumeData.js';
import type { StreamingVolumeData } from '../data/StreamingVolumeData.js';
import { loadRaw, loadRawStreaming } from './RawLoader.js';
import { loadDicom } from './DicomLoader.js';
import { loadNifti } from './NiftiLoader.js';
import { loadTiff } from './TiffLoader.js';
import { loadImage } from './ImageLoader.js';

/** Progress callback for loading stages */
export type LoadProgress = (stage: string, pct?: number) => void;

/** Called when a full-resolution volume replaces a streaming/preview volume */
export type VolumeSwapCallback = (volume: VolumeData) => void;

// Size thresholds for loading modes
const HYBRID_THRESHOLD = 200 * 1024 * 1024;   // 200MB
const STREAMING_THRESHOLD = 2 * 1024 * 1024 * 1024; // 2GB

/**
 * Dispatch a FileGroup to the appropriate loader.
 *
 * Three modes for RAW files based on size:
 *   - Standard  (< 200MB): Load full volume into memory
 *   - Hybrid    (200MB–2GB): Stream 4x preview first, load full in background, swap
 *   - Streaming (> 2GB): Never load full data, on-demand slices from File reference
 *
 * Non-RAW formats always use standard (full load).
 */
export async function loadVolume(
    group: FileGroup,
    onProgress?: LoadProgress,
    onVolumeSwap?: VolumeSwapCallback,
): Promise<VolumeData | StreamingVolumeData> {
    switch (group.type) {
        case '3d-raw': {
            if (!group.metadataFile) throw new Error('RAW file has no paired metadata file');
            const rawFile = group.files[0];
            const metaFile = group.metadataFile;
            const size = rawFile.size;

            if (size > STREAMING_THRESHOLD) {
                // Streaming mode: never load full file
                return loadRawStreaming(rawFile, metaFile, onProgress);
            }

            if (size > HYBRID_THRESHOLD) {
                // Hybrid mode: fast preview first, then try full load in background
                const streaming = await loadRawStreaming(rawFile, metaFile, onProgress);

                // Kick off background full load
                if (onVolumeSwap) {
                    loadFullInBackground(rawFile, metaFile, onProgress, onVolumeSwap);
                }

                return streaming;
            }

            // Standard mode: load directly into memory
            return loadRaw(rawFile, metaFile, onProgress);
        }
        case 'dicom':
            onProgress?.('Loading DICOM series...');
            return loadDicom(group);
        case 'nifti':
            onProgress?.('Loading NIfTI...');
            return loadNifti(group.files[0]);
        case 'tiff':
            onProgress?.('Loading TIFF...');
            return loadTiff(group.files[0]);
        case '2d-image':
            onProgress?.('Loading image...');
            return loadImage(group.files[0]);
        default:
            throw new Error(`Unknown file group type: ${group.type}`);
    }
}

/** Try to load the full volume in the background; call onSwap on success, silently fail on OOM */
function loadFullInBackground(
    rawFile: File,
    metaFile: File,
    onProgress: LoadProgress | undefined,
    onSwap: VolumeSwapCallback,
): void {
    (async () => {
        try {
            onProgress?.('Loading full resolution...');
            const full = await loadRaw(rawFile, metaFile, (stage, pct) => {
                onProgress?.(`Full: ${stage}`, pct);
            });
            onSwap(full);
        } catch (err) {
            console.warn('Background full-volume load failed (keeping streaming):', err);
        }
    })();
}
