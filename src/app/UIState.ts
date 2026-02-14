import { AppMode, type ViewAxis, type ViewId, type SegmentationSettings } from '../types.js';

/** Current viewer state */
export interface ViewerState {
    fileName: string;
    zoom: number;
    panX: number;
    panY: number;
    slices: Record<ViewAxis, number>;
    activeView: ViewId | null;
    crosshairsEnabled: boolean;
    appMode: AppMode;
    loading: boolean;
    gpuAvailable: boolean;
    segmentation: SegmentationSettings;
}

/** Events emitted by the UI state manager */
export interface UIEventMap {
    slicechange: { axis: ViewAxis; index: number };
    zoomchange: { zoom: number };
    fileloaded: { fileName: string };
    loading: { loading: boolean };
    statechange: { state: ViewerState };
}

type EventHandler<T> = (data: T) => void;
type Unsubscribe = () => void;

/**
 * Typed event emitter + state container for viewer UI.
 */
export class UIState {
    private _state: ViewerState;
    private _listeners = new Map<string, Set<EventHandler<unknown>>>();

    constructor() {
        this._state = {
            fileName: 'No file loaded',
            zoom: 1.0,
            panX: 0,
            panY: 0,
            slices: { xy: 0, xz: 0, yz: 0 },
            activeView: null,
            crosshairsEnabled: false,
            appMode: AppMode.Viewing,
            loading: false,
            gpuAvailable: false,
            segmentation: {
                enabled: false,
                visible: true,
                overlayOpacity: 0.4,
                color: [1.0, 0.0, 0.0],
                activeROIId: null,
                activeTool: 'brush',
                showOnlyActive: false,
                aiPreviewMask: null,
                isPinned: false,
                activeClassId: 1,
                tool: 'brush',
                brushRadius: 8,
                paintValue: 1,
                thresholdMin: 0,
                thresholdMax: 0,
                regionGrowTolerance: 25,
                regionGrowSliceRadius: 1,
            },
        };
    }

    /** Current state (read-only snapshot) */
    get state(): Readonly<ViewerState> {
        return this._state;
    }

    /** Subscribe to a typed event. Returns an unsubscribe function. */
    on<K extends keyof UIEventMap>(event: K, fn: EventHandler<UIEventMap[K]>): Unsubscribe {
        const key = event as string;
        if (!this._listeners.has(key)) {
            this._listeners.set(key, new Set());
        }
        const set = this._listeners.get(key)!;
        set.add(fn as EventHandler<unknown>);
        return () => { set.delete(fn as EventHandler<unknown>); };
    }

    /** Emit an event to all listeners */
    private emit<K extends keyof UIEventMap>(event: K, data: UIEventMap[K]): void {
        const set = this._listeners.get(event as string);
        if (set) {
            for (const fn of set) {
                fn(data);
            }
        }
    }

    /** Merge a partial state update and emit statechange */
    update(patch: Partial<ViewerState>): void {
        Object.assign(this._state, patch);
        this.emit('statechange', { state: this._state });
    }

    /** Update a specific slice index and emit slicechange */
    setSlice(axis: ViewAxis, index: number): void {
        this._state.slices[axis] = index;
        this.emit('slicechange', { axis, index });
        this.emit('statechange', { state: this._state });
    }

    /** Update zoom and emit zoomchange */
    setZoom(zoom: number): void {
        this._state.zoom = zoom;
        this.emit('zoomchange', { zoom });
        this.emit('statechange', { state: this._state });
    }

    /** Set file loaded and emit fileloaded */
    setFileLoaded(fileName: string): void {
        this._state.fileName = fileName;
        this.emit('fileloaded', { fileName });
        this.emit('statechange', { state: this._state });
    }

    /** Set loading state and emit loading */
    setLoading(loading: boolean): void {
        this._state.loading = loading;
        this.emit('loading', { loading });
        this.emit('statechange', { state: this._state });
    }
}
