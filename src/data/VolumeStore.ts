import type { StreamingVolumeData } from './StreamingVolumeData.js';
import type { VolumeData } from './VolumeData.js';

export type ViewerVolume = VolumeData | StreamingVolumeData;
export type ThreeDResolution = 'low' | 'mid' | 'full';

const DEFAULT_3D_CACHE_BUDGET_BYTES = 512 * 1024 * 1024; // 512MB
const MIN_3D_CACHE_BUDGET_BYTES = 32 * 1024 * 1024; // 32MB

interface ReplaceVolumeOptions {
    disposePreviousStreaming?: boolean;
    reset3DCache?: boolean;
    resetResolution?: boolean;
}

interface VolumeStoreOptions {
    cacheBudgetBytes?: number;
}

interface CacheInsertResult {
    cached: boolean;
    evicted: ThreeDResolution[];
    usageBytes: number;
}

function clampCacheBudgetBytes(value: number): number {
    if (!Number.isFinite(value)) return DEFAULT_3D_CACHE_BUDGET_BYTES;
    return Math.max(MIN_3D_CACHE_BUDGET_BYTES, Math.floor(value));
}

/**
 * Central owner for currently loaded volume and 3D resolution cache state.
 * Keeps volume lifecycle and cache policy separate from UI orchestration.
 */
export class VolumeStore {
    private _volume: ViewerVolume | null = null;
    private _current3DResolution: ThreeDResolution = 'low';
    private _cached3DVolumes = new Map<ThreeDResolution, VolumeData>();
    private _cached3DUsageBytes = 0;
    private _cacheBudgetBytes: number;

    constructor(options: VolumeStoreOptions = {}) {
        this._cacheBudgetBytes = clampCacheBudgetBytes(options.cacheBudgetBytes ?? DEFAULT_3D_CACHE_BUDGET_BYTES);
    }

    get volume(): ViewerVolume | null {
        return this._volume;
    }

    set volume(next: ViewerVolume | null) {
        this._volume = next;
    }

    get current3DResolution(): ThreeDResolution {
        return this._current3DResolution;
    }

    set current3DResolution(next: ThreeDResolution) {
        this._current3DResolution = next;
    }

    get3DCacheBudgetBytes(): number {
        return this._cacheBudgetBytes;
    }

    set3DCacheBudgetBytes(nextBytes: number): void {
        this._cacheBudgetBytes = clampCacheBudgetBytes(nextBytes);
        this.enforceCacheBudget();
    }

    get3DCacheUsageBytes(): number {
        return this._cached3DUsageBytes;
    }

    getCached3DVolume(resolution: ThreeDResolution): VolumeData | null {
        const cached = this._cached3DVolumes.get(resolution);
        if (!cached) return null;
        // Refresh insertion order to preserve LRU semantics.
        this._cached3DVolumes.delete(resolution);
        this._cached3DVolumes.set(resolution, cached);
        return cached;
    }

    cache3DVolume(resolution: ThreeDResolution, volume: VolumeData): CacheInsertResult {
        const bytes = this.estimateCachedVolumeBytes(volume);
        const evicted: ThreeDResolution[] = [];

        this.removeCached3DVolume(resolution);
        if (bytes > this._cacheBudgetBytes) {
            return { cached: false, evicted, usageBytes: this._cached3DUsageBytes };
        }

        this._cached3DVolumes.set(resolution, volume);
        this._cached3DUsageBytes += bytes;
        evicted.push(...this.enforceCacheBudget(resolution));
        return { cached: true, evicted, usageBytes: this._cached3DUsageBytes };
    }

    clear3DCache(): void {
        this._cached3DVolumes.clear();
        this._cached3DUsageBytes = 0;
    }

    private estimateCachedVolumeBytes(volume: VolumeData): number {
        return volume.data.byteLength;
    }

    private removeCached3DVolume(resolution: ThreeDResolution): boolean {
        const cached = this._cached3DVolumes.get(resolution);
        if (!cached) return false;
        this._cached3DUsageBytes = Math.max(
            0,
            this._cached3DUsageBytes - this.estimateCachedVolumeBytes(cached),
        );
        this._cached3DVolumes.delete(resolution);
        return true;
    }

    private enforceCacheBudget(protectedResolution?: ThreeDResolution): ThreeDResolution[] {
        const evicted: ThreeDResolution[] = [];
        while (this._cached3DUsageBytes > this._cacheBudgetBytes && this._cached3DVolumes.size > 0) {
            const oldest = this._cached3DVolumes.keys().next().value as ThreeDResolution | undefined;
            if (!oldest) break;
            if (protectedResolution && oldest === protectedResolution && this._cached3DVolumes.size > 1) {
                const keep = this._cached3DVolumes.get(oldest);
                if (!keep) break;
                this._cached3DVolumes.delete(oldest);
                this._cached3DVolumes.set(oldest, keep);
                continue;
            }
            if (this.removeCached3DVolume(oldest)) {
                evicted.push(oldest);
            } else {
                break;
            }
        }
        return evicted;
    }

    replaceVolume(next: ViewerVolume | null, options: ReplaceVolumeOptions = {}): void {
        const disposePreviousStreaming = options.disposePreviousStreaming ?? false;
        const reset3DCache = options.reset3DCache ?? false;
        const resetResolution = options.resetResolution ?? false;

        if (disposePreviousStreaming && this._volume && this._volume !== next && this._volume.isStreaming) {
            this._volume.dispose();
        }
        this._volume = next;

        if (reset3DCache) {
            this.clear3DCache();
        }
        if (resetResolution) {
            this._current3DResolution = 'low';
        }
    }

    clear(options: { disposeStreaming?: boolean; resetResolution?: boolean } = {}): void {
        const disposeStreaming = options.disposeStreaming ?? true;
        const resetResolution = options.resetResolution ?? true;
        if (disposeStreaming && this._volume && this._volume.isStreaming) {
            this._volume.dispose();
        }
        this._volume = null;
        this.clear3DCache();
        if (resetResolution) {
            this._current3DResolution = 'low';
        }
    }
}
