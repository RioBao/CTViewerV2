type CrashLogLevel = 'event' | 'warning' | 'error' | 'fatal';

interface CrashSessionState {
    version: 1;
    sessionId: string;
    startedAt: string;
    lastHeartbeatAt: string;
    closed: boolean;
    endedAt?: string;
    page: string;
    userAgent: string;
}

interface CrashLogEntry {
    version: 1;
    sessionId: string;
    level: CrashLogLevel;
    timestamp: string;
    message: string;
}

interface CrashLogStore {
    version: 1;
    entries: CrashLogEntry[];
}

interface CrashLogApi {
    getEntries: () => CrashLogEntry[];
    clear: () => void;
}

declare global {
    interface Window {
        viewerCrashLog?: CrashLogApi;
    }
}

const CRASH_LOG_ENTRIES_KEY = 'viewerV2.crashlog.entries';
const CRASH_LOG_SESSION_KEY = 'viewerV2.crashlog.session';
const CRASH_LOG_MAX_ENTRIES = 250;
const HEARTBEAT_MS = 5000;

let initialized = false;
let currentSessionId = '';
let heartbeatTimer: number | null = null;
let markClosedBound = () => {};
let onErrorBound = (_event: ErrorEvent) => {};
let onUnhandledRejectionBound = (_event: PromiseRejectionEvent) => {};
let originalConsoleError: ((...data: unknown[]) => void) | null = null;
let originalConsoleWarn: ((...data: unknown[]) => void) | null = null;

function nowIso(): string {
    return new Date().toISOString();
}

function readSessionState(): CrashSessionState | null {
    try {
        const raw = localStorage.getItem(CRASH_LOG_SESSION_KEY);
        if (!raw) return null;
        const parsed = JSON.parse(raw) as Partial<CrashSessionState>;
        if (!parsed || parsed.version !== 1 || typeof parsed.sessionId !== 'string') return null;
        return parsed as CrashSessionState;
    } catch {
        return null;
    }
}

function writeSessionState(state: CrashSessionState): void {
    try {
        localStorage.setItem(CRASH_LOG_SESSION_KEY, JSON.stringify(state));
    } catch {
        // Ignore storage failures.
    }
}

function readStore(): CrashLogStore {
    try {
        const raw = localStorage.getItem(CRASH_LOG_ENTRIES_KEY);
        if (!raw) return { version: 1, entries: [] };
        const parsed = JSON.parse(raw) as Partial<CrashLogStore>;
        if (!parsed || parsed.version !== 1 || !Array.isArray(parsed.entries)) {
            return { version: 1, entries: [] };
        }
        return { version: 1, entries: parsed.entries as CrashLogEntry[] };
    } catch {
        return { version: 1, entries: [] };
    }
}

function writeStore(store: CrashLogStore): void {
    try {
        localStorage.setItem(CRASH_LOG_ENTRIES_KEY, JSON.stringify(store));
    } catch {
        // Ignore storage failures.
    }
}

function pushEntry(level: CrashLogLevel, message: string): void {
    const store = readStore();
    store.entries.push({
        version: 1,
        sessionId: currentSessionId,
        level,
        timestamp: nowIso(),
        message: message.slice(0, 4000),
    });
    if (store.entries.length > CRASH_LOG_MAX_ENTRIES) {
        store.entries.splice(0, store.entries.length - CRASH_LOG_MAX_ENTRIES);
    }
    writeStore(store);
}

function safeStringify(value: unknown): string {
    if (value instanceof Error) {
        return `${value.name}: ${value.message}${value.stack ? `\n${value.stack}` : ''}`;
    }
    if (typeof value === 'string') return value;
    if (typeof value === 'number' || typeof value === 'boolean' || value == null) {
        return String(value);
    }
    try {
        return JSON.stringify(value);
    } catch {
        return Object.prototype.toString.call(value);
    }
}

function formatConsoleArgs(args: unknown[]): string {
    if (args.length === 0) return '(no args)';
    return args.map((a) => safeStringify(a)).join(' | ');
}

function patchConsole(): void {
    if (!originalConsoleError) {
        originalConsoleError = console.error.bind(console);
    }
    if (!originalConsoleWarn) {
        originalConsoleWarn = console.warn.bind(console);
    }

    console.error = (...data: unknown[]) => {
        pushEntry('error', `console.error: ${formatConsoleArgs(data)}`);
        originalConsoleError?.(...data);
    };

    console.warn = (...data: unknown[]) => {
        pushEntry('warning', `console.warn: ${formatConsoleArgs(data)}`);
        originalConsoleWarn?.(...data);
    };
}

function unpatchConsole(): void {
    if (originalConsoleError) {
        console.error = originalConsoleError;
    }
    if (originalConsoleWarn) {
        console.warn = originalConsoleWarn;
    }
}

function updateHeartbeat(): void {
    const state = readSessionState();
    if (!state || state.sessionId !== currentSessionId || state.closed) return;
    state.lastHeartbeatAt = nowIso();
    writeSessionState(state);
}

function markSessionClosed(): void {
    const state = readSessionState();
    if (!state || state.sessionId !== currentSessionId || state.closed) return;
    state.closed = true;
    state.endedAt = nowIso();
    state.lastHeartbeatAt = state.endedAt;
    writeSessionState(state);
}

export function initCrashLogger(): void {
    if (initialized) return;
    initialized = true;
    currentSessionId = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;

    const previousSession = readSessionState();
    if (previousSession && !previousSession.closed) {
        pushEntry(
            'fatal',
            `Detected unclean shutdown in previous session (${previousSession.sessionId}) at ${previousSession.lastHeartbeatAt}`,
        );
    }

    const startedAt = nowIso();
    writeSessionState({
        version: 1,
        sessionId: currentSessionId,
        startedAt,
        lastHeartbeatAt: startedAt,
        closed: false,
        page: location.href,
        userAgent: navigator.userAgent,
    });

    pushEntry('event', `Session started (${currentSessionId})`);
    patchConsole();

    onErrorBound = (event: ErrorEvent) => {
        const message = event.error instanceof Error
            ? `${event.error.name}: ${event.error.message}${event.error.stack ? `\n${event.error.stack}` : ''}`
            : `${event.message} @ ${event.filename}:${event.lineno}:${event.colno}`;
        pushEntry('fatal', `window.error: ${message}`);
    };
    window.addEventListener('error', onErrorBound);

    onUnhandledRejectionBound = (event: PromiseRejectionEvent) => {
        pushEntry('fatal', `unhandledrejection: ${safeStringify(event.reason)}`);
    };
    window.addEventListener('unhandledrejection', onUnhandledRejectionBound);

    markClosedBound = () => markSessionClosed();
    window.addEventListener('beforeunload', markClosedBound);
    window.addEventListener('pagehide', markClosedBound);

    heartbeatTimer = window.setInterval(updateHeartbeat, HEARTBEAT_MS);

    window.viewerCrashLog = {
        getEntries: () => readStore().entries,
        clear: () => {
            writeStore({ version: 1, entries: [] });
        },
    };
}

export function disposeCrashLogger(): void {
    if (!initialized) return;
    initialized = false;
    if (heartbeatTimer != null) {
        clearInterval(heartbeatTimer);
        heartbeatTimer = null;
    }
    markSessionClosed();
    window.removeEventListener('beforeunload', markClosedBound);
    window.removeEventListener('pagehide', markClosedBound);
    window.removeEventListener('error', onErrorBound);
    window.removeEventListener('unhandledrejection', onUnhandledRejectionBound);
    unpatchConsole();
}

