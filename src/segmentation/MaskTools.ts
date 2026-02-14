import type { ViewAxis } from '../types.js';
import type { MaskVolume } from './MaskTypes.js';

const circleKernelCache = new Map<number, Array<[number, number]>>();

function clampClassId(classId: number, maxClassId: number): number {
    if (!Number.isFinite(classId)) return 0;
    const rounded = Math.round(classId);
    if (rounded < 0) return 0;
    if (rounded > maxClassId) return maxClassId;
    return rounded;
}

function getCircleOffsets(radius: number): Array<[number, number]> {
    const r = Math.max(1, Math.floor(radius));
    const cached = circleKernelCache.get(r);
    if (cached) return cached;

    const rr = r * r;
    const offsets: Array<[number, number]> = [];
    for (let dy = -r; dy <= r; dy++) {
        for (let dx = -r; dx <= r; dx++) {
            if (dx * dx + dy * dy <= rr) {
                offsets.push([dx, dy]);
            }
        }
    }

    circleKernelCache.set(r, offsets);
    return offsets;
}

function axisSliceBounds(axis: ViewAxis, dimensions: [number, number, number]): number {
    switch (axis) {
        case 'xy': return dimensions[2];
        case 'xz': return dimensions[1];
        case 'yz': return dimensions[0];
    }
}

function mapSliceToVoxel(
    axis: ViewAxis,
    sliceIndex: number,
    sliceX: number,
    sliceY: number,
): [number, number, number] {
    switch (axis) {
        case 'xy': return [sliceX, sliceY, sliceIndex];
        case 'xz': return [sliceX, sliceIndex, sliceY];
        case 'yz': return [sliceIndex, sliceX, sliceY];
    }
}

/**
 * Stamp a circular brush into a segmentation mask.
 * Multi-class friendly via classId (binary uses 0/1).
 */
export function stampBrush2p5D(
    mask: MaskVolume,
    axis: ViewAxis,
    sliceIndex: number,
    sliceX: number,
    sliceY: number,
    radius: number,
    classId: number,
    sliceRadius = 0,
    onVoxelChanged?: (x: number, y: number, z: number, previousClassId: number, nextClassId: number) => void,
): number {
    const cx = Math.floor(sliceX);
    const cy = Math.floor(sliceY);
    const offsets = getCircleOffsets(radius);
    const maxSlices = axisSliceBounds(axis, mask.dimensions);
    const clampedSliceRadius = Math.max(0, Math.floor(sliceRadius));
    const nextClassId = clampClassId(classId, mask.maxClassId);

    let changed = 0;
    for (let ds = -clampedSliceRadius; ds <= clampedSliceRadius; ds++) {
        const s = sliceIndex + ds;
        if (s < 0 || s >= maxSlices) continue;
        for (const [dx, dy] of offsets) {
            const [x, y, z] = mapSliceToVoxel(axis, s, cx + dx, cy + dy);
            const previousClassId = mask.getVoxel(x, y, z);
            if (previousClassId === nextClassId) continue;
            if (mask.setVoxel(x, y, z, nextClassId)) {
                changed++;
                onVoxelChanged?.(x, y, z, previousClassId, nextClassId);
            }
        }
    }
    return changed;
}
