import type { MaskTypedArray, ViewAxis } from '../../types.js';
import type { SegmentationOp, SegmentationOpContext, SegmentationOpResult } from '../OpsQueue.js';

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

    // Compact flat-array undo data — avoids allocating one JS object per changed voxel.
    private undoLinear: Uint32Array = new Uint32Array(0);
    private undoBefore: MaskTypedArray = new Uint8Array(0);

    private readonly axis: ViewAxis;
    private readonly sliceIndex: number;
    private readonly width: number;
    private readonly selectedIndices: Uint32Array;
    private readonly classId: number;

    constructor(params: ApplySliceSelectionOpParams) {
        this.axis = params.axis;
        this.sliceIndex = params.sliceIndex;
        this.width = Math.max(1, Math.floor(params.width));
        this.selectedIndices = params.selectedIndices;
        this.classId = params.classId;
    }

    apply(ctx: SegmentationOpContext): SegmentationOpResult {
        if (!this.committed) {
            // First apply: use the bulk write path which does direct array access,
            // calls markAllDirty once, and returns compact flat-array undo data.
            const result = ctx.mask.writeSliceSelection(
                this.axis, this.sliceIndex, this.width, this.selectedIndices, this.classId,
            );
            this.committed = true;
            this.undoLinear = result.undoLinear;
            this.undoBefore = result.undoBefore;
            return { changedVoxels: result.changed };
        }

        // Redo: restore the same voxels to classId using compact linear indices.
        const redoBefore = new (this.undoBefore.constructor as { new(n: number): MaskTypedArray })(this.undoLinear.length);
        redoBefore.fill(this.classId);
        const changed = ctx.mask.restoreLinearValues(this.undoLinear, redoBefore);
        return { changedVoxels: changed };
    }

    undo(ctx: SegmentationOpContext): SegmentationOpResult {
        const changed = ctx.mask.restoreLinearValues(this.undoLinear, this.undoBefore);
        return { changedVoxels: changed };
    }
}

export function createApplySliceSelectionOp(params: ApplySliceSelectionOpParams): SegmentationOp {
    return new ApplySliceSelectionOp(params);
}
