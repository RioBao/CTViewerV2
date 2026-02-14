import type { ViewAxis } from '../../types.js';
import { stampBrush2p5D } from '../MaskTools.js';
import type { SegmentationOp, SegmentationOpContext, SegmentationOpResult } from '../OpsQueue.js';

interface VoxelDelta {
    x: number;
    y: number;
    z: number;
    before: number;
    after: number;
}

function voxelKey(x: number, y: number, z: number): string {
    return `${x},${y},${z}`;
}

export interface PaintStrokeOpParams {
    axis: ViewAxis;
    sliceIndex: number;
    sliceX: number;
    sliceY: number;
    radius: number;
    classId: number;
    sliceRadius?: number;
    mergeKey?: string;
    onVoxelChanged?: (x: number, y: number, z: number, previousClassId: number, nextClassId: number) => void;
}

class PaintStrokeOp implements SegmentationOp {
    readonly type = 'paint-stroke';
    readonly mergeKey?: string;
    private committed = false;
    private deltas: VoxelDelta[] = [];
    private deltaIndex = new Map<string, number>();
    private onVoxelChanged?: PaintStrokeOpParams['onVoxelChanged'];

    private readonly axis: ViewAxis;
    private readonly sliceIndex: number;
    private readonly sliceX: number;
    private readonly sliceY: number;
    private readonly radius: number;
    private readonly classId: number;
    private readonly sliceRadius: number;

    constructor(params: PaintStrokeOpParams) {
        this.axis = params.axis;
        this.sliceIndex = params.sliceIndex;
        this.sliceX = params.sliceX;
        this.sliceY = params.sliceY;
        this.radius = params.radius;
        this.classId = params.classId;
        this.sliceRadius = Math.max(0, Math.floor(params.sliceRadius ?? 0));
        this.mergeKey = params.mergeKey;
        this.onVoxelChanged = params.onVoxelChanged;
    }

    apply(ctx: SegmentationOpContext): SegmentationOpResult {
        if (!this.committed) {
            stampBrush2p5D(
                ctx.mask,
                this.axis,
                this.sliceIndex,
                this.sliceX,
                this.sliceY,
                this.radius,
                this.classId,
                this.sliceRadius,
                (x, y, z, previousClassId, nextClassId) => {
                    const key = voxelKey(x, y, z);
                    const existing = this.deltaIndex.get(key);
                    if (existing == null) {
                        this.deltaIndex.set(key, this.deltas.length);
                        this.deltas.push({ x, y, z, before: previousClassId, after: nextClassId });
                    } else {
                        this.deltas[existing].after = nextClassId;
                    }
                    this.onVoxelChanged?.(x, y, z, previousClassId, nextClassId);
                },
            );
            this.committed = true;
            return { changedVoxels: this.deltas.length };
        }

        let changed = 0;
        for (const delta of this.deltas) {
            const previous = ctx.mask.getVoxel(delta.x, delta.y, delta.z);
            if (previous === delta.after) continue;
            if (ctx.mask.setVoxel(delta.x, delta.y, delta.z, delta.after)) {
                changed++;
                this.onVoxelChanged?.(delta.x, delta.y, delta.z, previous, delta.after);
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
                this.onVoxelChanged?.(delta.x, delta.y, delta.z, previous, delta.before);
            }
        }
        return { changedVoxels: changed };
    }

    merge(next: SegmentationOp): void {
        if (!(next instanceof PaintStrokeOp)) return;
        for (const delta of next.deltas) {
            const key = voxelKey(delta.x, delta.y, delta.z);
            const existing = this.deltaIndex.get(key);
            if (existing == null) {
                this.deltaIndex.set(key, this.deltas.length);
                this.deltas.push({ ...delta });
            } else {
                this.deltas[existing].after = delta.after;
            }
        }
    }
}

export function createPaintStrokeOp(params: PaintStrokeOpParams): SegmentationOp {
    return new PaintStrokeOp(params);
}
