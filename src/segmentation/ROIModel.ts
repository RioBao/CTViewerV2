export interface ROIEntry {
    id: string;
    classId: number;
    name: string;
    colorHex: string;
    visible: boolean;
    locked: boolean;
    aiBusy: boolean;
}

export interface ROIStats {
    voxels: number;
    volume: number;
    meanIntensity: number;
    stdIntensity: number;
}

const ROI_COLOR_PALETTE = [
    '#ff4d4f',
    '#ff9f1a',
    '#ffd60a',
    '#52c41a',
    '#16c79a',
    '#36cfc9',
    '#40a9ff',
    '#597ef7',
    '#9254de',
    '#eb2f96',
];

export function roiColorForIndex(index: number): string {
    const safe = Math.max(0, Math.floor(index));
    return ROI_COLOR_PALETTE[safe % ROI_COLOR_PALETTE.length];
}

export function createROIEntry(id: string, classId: number, ordinal: number): ROIEntry {
    return {
        id,
        classId,
        name: `ROI ${ordinal}`,
        colorHex: roiColorForIndex(ordinal - 1),
        visible: true,
        locked: false,
        aiBusy: false,
    };
}

export function hexToRgb01(hex: string): [number, number, number] {
    const normalized = hex.replace('#', '').trim();
    if (!/^[0-9a-fA-F]{6}$/.test(normalized)) {
        return [1.0, 0.0, 0.0];
    }
    return [
        parseInt(normalized.slice(0, 2), 16) / 255,
        parseInt(normalized.slice(2, 4), 16) / 255,
        parseInt(normalized.slice(4, 6), 16) / 255,
    ];
}
