import type { MaskVolume } from './MaskTypes.js';

export interface SegmentationOpContext {
    mask: MaskVolume;
}

export interface SegmentationOpResult {
    changedVoxels: number;
}

export interface SegmentationOp {
    readonly type: string;
    readonly mergeKey?: string;
    apply(ctx: SegmentationOpContext): SegmentationOpResult;
    undo(ctx: SegmentationOpContext): SegmentationOpResult;
    merge?(next: SegmentationOp): void;
}

const NO_CHANGE: SegmentationOpResult = { changedVoxels: 0 };

/**
 * FIFO-style operation queue with bounded undo/redo stacks.
 * All segmentation edits should flow through this queue.
 */
export class OpsQueue {
    private undoStack: SegmentationOp[] = [];
    private redoStack: SegmentationOp[] = [];

    constructor(private readonly maxDepth = 512) {}

    apply(op: SegmentationOp, ctx: SegmentationOpContext): SegmentationOpResult {
        const result = op.apply(ctx);
        if (result.changedVoxels <= 0) {
            return NO_CHANGE;
        }

        let merged = false;
        const top = this.undoStack[this.undoStack.length - 1];
        if (top && top.merge && top.mergeKey && op.mergeKey && top.mergeKey === op.mergeKey) {
            top.merge(op);
            merged = true;
        }

        if (!merged) {
            this.undoStack.push(op);
            if (this.undoStack.length > this.maxDepth) {
                this.undoStack.shift();
            }
        }

        this.redoStack.length = 0;
        return result;
    }

    undo(ctx: SegmentationOpContext): SegmentationOpResult {
        const op = this.undoStack.pop();
        if (!op) return NO_CHANGE;
        const result = op.undo(ctx);
        if (result.changedVoxels > 0) {
            this.redoStack.push(op);
            return result;
        }
        return NO_CHANGE;
    }

    redo(ctx: SegmentationOpContext): SegmentationOpResult {
        const op = this.redoStack.pop();
        if (!op) return NO_CHANGE;
        const result = op.apply(ctx);
        if (result.changedVoxels > 0) {
            this.undoStack.push(op);
            if (this.undoStack.length > this.maxDepth) {
                this.undoStack.shift();
            }
            return result;
        }
        return NO_CHANGE;
    }

    clear(): void {
        this.undoStack.length = 0;
        this.redoStack.length = 0;
    }

    canUndo(): boolean {
        return this.undoStack.length > 0;
    }

    canRedo(): boolean {
        return this.redoStack.length > 0;
    }
}
