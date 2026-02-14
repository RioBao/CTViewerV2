import type { ViewAxis } from '../../types.js';
import type { SegmentationOp, SegmentationOpContext, SegmentationOpResult } from '../OpsQueue.js';

interface SliceDelta {
    x: number;
    y: number;
    z: number;
    before: number;
    after: number;
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

export interface ApplySliceSelectionOpParams {
    axis: ViewAxis;
    sliceIndex: number;
    width: number;
    height: number;
    selectedIndices: Uint32Array;
    classId: number;
}

class ApplySliceSelectionOp implements SegmentationOp {
    readonly type = 'apply-slice-selection';
    private committed = false;
    private deltas: SliceDelta[] = [];

    private readonly axis: ViewAxis;
    private readonly sliceIndex: number;
    private readonly width: number;
    private readonly height: number;
    private readonly selectedIndices: Uint32Array;
    private readonly classId: number;

    constructor(params: ApplySliceSelectionOpParams) {
        this.axis = params.axis;
        this.sliceIndex = params.sliceIndex;
        this.width = Math.max(1, Math.floor(params.width));
        this.height = Math.max(1, Math.floor(params.height));
        this.selectedIndices = params.selectedIndices;
        this.classId = params.classId;
    }

    apply(ctx: SegmentationOpContext): SegmentationOpResult {
        if (!this.committed) {
            let changed = 0;
            for (let i = 0; i < this.selectedIndices.length; i++) {
                const idx = this.selectedIndices[i];
                const sx = idx % this.width;
                const sy = Math.floor(idx / this.width);
                if (sx < 0 || sx >= this.width || sy < 0 || sy >= this.height) continue;
                const [x, y, z] = mapSliceToVoxel(this.axis, this.sliceIndex, sx, sy);
                const before = ctx.mask.getVoxel(x, y, z);
                if (before === this.classId) continue;
                if (ctx.mask.setVoxel(x, y, z, this.classId)) {
                    changed++;
                    this.deltas.push({ x, y, z, before, after: this.classId });
                }
            }
            this.committed = true;
            return { changedVoxels: changed };
        }

        let changed = 0;
        for (const delta of this.deltas) {
            const previous = ctx.mask.getVoxel(delta.x, delta.y, delta.z);
            if (previous === delta.after) continue;
            if (ctx.mask.setVoxel(delta.x, delta.y, delta.z, delta.after)) {
                changed++;
            }
        }
        return { changedVoxels: changed };
    }

    undo(ctx: SegmentationOpContext): SegmentationOpResult {
        let changed = 0;
        for (let i = this.deltas.length - 1; i >= 0; i--) {
            const delta = this.deltas[i];
            const previous = ctx.mask.getVoxel(delta.x, delta.y, delta.z);
            if (previous === delta.before) continue;
            if (ctx.mask.setVoxel(delta.x, delta.y, delta.z, delta.before)) {
                changed++;
            }
        }
        return { changedVoxels: changed };
    }
}

export function createApplySliceSelectionOp(params: ApplySliceSelectionOpParams): SegmentationOp {
    return new ApplySliceSelectionOp(params);
}
