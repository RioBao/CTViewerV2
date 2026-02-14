import { createROIEntry, type ROIEntry } from './ROIModel.js';

/**
 * Central registry for ROI metadata and class-id assignment.
 * Keeps ROI bookkeeping out of UI orchestration code.
 */
export class ROIRegistry {
    private _entries: ROIEntry[] = [];
    private idCounter = 1;
    private ordinalCounter = 1;

    get entries(): ROIEntry[] {
        return this._entries;
    }

    reset(): void {
        this._entries = [];
        this.idCounter = 1;
        this.ordinalCounter = 1;
    }

    findById(id: string | null): ROIEntry | null {
        if (!id) return null;
        return this._entries.find((roi) => roi.id === id) ?? null;
    }

    findByClassId(classId: number): ROIEntry | null {
        return this._entries.find((roi) => roi.classId === classId) ?? null;
    }

    nextAvailableClassId(maxClassId = 255): number | null {
        const used = new Set(this._entries.map((roi) => roi.classId));
        for (let classId = 1; classId <= maxClassId; classId++) {
            if (!used.has(classId)) return classId;
        }
        return null;
    }

    add(maxClassId = 255): ROIEntry | null {
        const classId = this.nextAvailableClassId(maxClassId);
        if (classId == null) return null;
        const roi = createROIEntry(`roi-${this.idCounter++}`, classId, this.ordinalCounter++);
        this._entries.push(roi);
        return roi;
    }

    addWithClassId(
        classId: number,
        options: {
            id?: string;
            name?: string;
            colorHex?: string;
            visible?: boolean;
            locked?: boolean;
        } = {},
    ): ROIEntry | null {
        if (!Number.isFinite(classId)) return null;
        const normalizedClassId = Math.max(1, Math.floor(classId));
        if (this.findByClassId(normalizedClassId)) return null;

        let id = options.id?.trim();
        if (!id) {
            id = `roi-${this.idCounter++}`;
        } else {
            if (this.findById(id)) return null;
            const match = /^roi-(\d+)$/i.exec(id);
            if (match) {
                const parsed = parseInt(match[1], 10);
                if (Number.isFinite(parsed)) {
                    this.idCounter = Math.max(this.idCounter, parsed + 1);
                }
            }
        }
        const roi = createROIEntry(id, normalizedClassId, this.ordinalCounter++);
        if (options.name) {
            roi.name = options.name;
        }
        if (options.colorHex) {
            roi.colorHex = options.colorHex;
        }
        if (options.visible !== undefined) {
            roi.visible = options.visible;
        }
        if (options.locked !== undefined) {
            roi.locked = options.locked;
        }
        this._entries.push(roi);
        return roi;
    }

    removeById(id: string): boolean {
        const prevLen = this._entries.length;
        this._entries = this._entries.filter((roi) => roi.id !== id);
        return this._entries.length !== prevLen;
    }
}
