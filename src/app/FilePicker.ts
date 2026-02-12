/** Supported file format extensions, matching old viewer config */
export const SUPPORTED_FORMATS = {
    image: ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp'],
    tiff: ['tif', 'tiff'],
    raw: ['raw'],
    rawMetadata: ['json', 'volumeinfo', 'dat'],
    dicom: ['dcm'],
    nifti: ['nii', 'nii.gz'],
} as const;

/** A grouped set of files ready for loading */
export interface FileGroup {
    type: '2d-image' | 'tiff' | '3d-raw' | 'dicom' | 'nifti';
    files: File[];
    name: string;
    /** For 3d-raw: paired metadata file */
    metadataFile?: File;
}

type FileHandler = (groups: FileGroup[]) => void;

/**
 * Manages file input via button click and drag-and-drop.
 * Groups selected files by type before handing them to the viewer.
 */
export class FilePicker {
    private fileInput: HTMLInputElement;
    private dropZone: HTMLElement;
    private onFiles: FileHandler;

    constructor(fileInput: HTMLInputElement, dropZone: HTMLElement, onFiles: FileHandler) {
        this.fileInput = fileInput;
        this.dropZone = dropZone;
        this.onFiles = onFiles;
        this.bind();
    }

    private bind(): void {
        // File input change
        this.fileInput.addEventListener('change', () => {
            if (this.fileInput.files && this.fileInput.files.length > 0) {
                const groups = groupFiles(Array.from(this.fileInput.files));
                this.onFiles(groups);
                this.fileInput.value = '';
            }
        });

        // Drag-and-drop
        this.dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.dropZone.classList.add('drag-over');
        });

        this.dropZone.addEventListener('dragleave', (e) => {
            if (!this.dropZone.contains(e.relatedTarget as Node)) {
                this.dropZone.classList.remove('drag-over');
            }
        });

        this.dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.dropZone.classList.remove('drag-over');
            if (e.dataTransfer?.files && e.dataTransfer.files.length > 0) {
                const groups = groupFiles(Array.from(e.dataTransfer.files));
                this.onFiles(groups);
            }
        });
    }

    /** Programmatically open the file picker dialog */
    open(): void {
        this.fileInput.click();
    }
}

// ---------- File grouping logic (ported from old FileParser.js) ----------

function getExtension(filename: string): string {
    const lower = filename.toLowerCase();
    // Handle .nii.gz compound extension
    if (lower.endsWith('.nii.gz')) return 'nii.gz';
    const parts = lower.split('.');
    return parts.length > 1 ? parts[parts.length - 1] : '';
}

function isNiftiFile(filename: string): boolean {
    const lower = filename.toLowerCase();
    return lower.endsWith('.nii') || lower.endsWith('.nii.gz');
}

/**
 * Group an array of files by type, pairing RAW files with their metadata.
 * Ported from old viewer's FileParser.groupFiles + groupFilesAsync (sync subset).
 */
export function groupFiles(files: File[]): FileGroup[] {
    const groups: FileGroup[] = [];
    const rawMap = new Map<string, { raw?: File; meta?: File }>();
    const processed = new Set<File>();

    // First pass: identify raw + metadata files
    for (const file of files) {
        const ext = getExtension(file.name);

        if (ext === 'raw') {
            const basename = file.name.replace(/\.raw$/i, '');
            if (!rawMap.has(basename)) rawMap.set(basename, {});
            rawMap.get(basename)!.raw = file;
            processed.add(file);
        } else if (ext === 'json') {
            const basename = file.name.replace(/\.json$/i, '');
            if (!rawMap.has(basename)) rawMap.set(basename, {});
            rawMap.get(basename)!.meta = file;
            processed.add(file);
        } else if (ext === 'volumeinfo') {
            // volumeinfo files: name.raw.volumeinfo
            const basename = file.name.replace(/\.raw\.volumeinfo$/i, '');
            if (!rawMap.has(basename)) rawMap.set(basename, {});
            const entry = rawMap.get(basename)!;
            if (!entry.meta) entry.meta = file; // json takes priority
            processed.add(file);
        } else if (ext === 'dat') {
            const basename = file.name.replace(/\.dat$/i, '');
            if (!rawMap.has(basename)) rawMap.set(basename, {});
            const entry = rawMap.get(basename)!;
            if (!entry.meta) entry.meta = file; // json/volumeinfo take priority
            processed.add(file);
        }
    }

    // Create 3d-raw groups from paired files
    for (const [basename, entry] of rawMap) {
        if (entry.raw && entry.meta) {
            groups.push({
                type: '3d-raw',
                files: [entry.raw],
                name: basename,
                metadataFile: entry.meta,
            });
        } else if (entry.raw) {
            console.warn(`RAW file ${basename}.raw found without matching metadata file`);
        } else if (entry.meta) {
            console.warn(`Metadata file for ${basename} found without matching RAW file`);
        }
    }

    // Second pass: remaining files
    const dicomFiles: File[] = [];

    for (const file of files) {
        if (processed.has(file)) continue;

        const ext = getExtension(file.name);

        if (isNiftiFile(file.name)) {
            groups.push({ type: 'nifti', files: [file], name: file.name });
        } else if (ext === 'dcm') {
            dicomFiles.push(file);
        } else if (SUPPORTED_FORMATS.tiff.includes(ext as typeof SUPPORTED_FORMATS.tiff[number])) {
            groups.push({ type: 'tiff', files: [file], name: file.name });
        } else if (SUPPORTED_FORMATS.image.includes(ext as typeof SUPPORTED_FORMATS.image[number])) {
            groups.push({ type: '2d-image', files: [file], name: file.name });
        }
    }

    // Group all DICOM files together as one series
    if (dicomFiles.length > 0) {
        groups.push({
            type: 'dicom',
            files: dicomFiles,
            name: `DICOM series (${dicomFiles.length} files)`,
        });
    }

    return groups;
}
