struct RegionGrowParams {
    width: u32,
    height: u32,
    total: u32,
    tolerance: f32,
    seedValue: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct RegionGrowState {
    readHead: atomic<u32>,
    writeTail: atomic<u32>,
    activeWorkers: atomic<u32>,
    _pad: atomic<u32>,
}

@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read_write> visited: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> queue: array<u32>;
@group(0) @binding(3) var<storage, read_write> state: RegionGrowState;
@group(0) @binding(4) var<storage, read> params: RegionGrowParams;

const WORKGROUP_SIZE: u32 = 256u;

fn try_enqueue(idx: u32) {
    let previous = atomicCompareExchangeWeak(&visited[idx], 0u, 1u);
    if (previous.old_value != 0u) {
        return;
    }

    let delta = abs(values[idx] - params.seedValue);
    if (!(delta <= params.tolerance)) {
        return;
    }

    let writeIndex = atomicAdd(&state.writeTail, 1u);
    queue[writeIndex] = idx;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn region_grow_persistent() {
    loop {
        let head = atomicLoad(&state.readHead);
        let tail = atomicLoad(&state.writeTail);

        if (head < tail) {
            // Signal active intent BEFORE claiming so the termination check
            // never sees activeWorkers==0 while a thread is between CAS and
            // processing (which would cause premature exit and serial fallback).
            atomicAdd(&state.activeWorkers, 1u);

            let claim = atomicCompareExchangeWeak(&state.readHead, head, head + 1u);
            if (claim.old_value != head) {
                // CAS failed — another thread claimed this slot; undo and retry.
                atomicSub(&state.activeWorkers, 1u);
                continue;
            }

            let idx = queue[head];
            let x = idx % params.width;
            let y = idx / params.width;

            for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
                for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
                    if (dx == 0 && dy == 0) {
                        continue;
                    }
                    let nx = i32(x) + dx;
                    let ny = i32(y) + dy;
                    if (nx < 0 || ny < 0 || nx >= i32(params.width) || ny >= i32(params.height)) {
                        continue;
                    }
                    let nidx = u32(ny) * params.width + u32(nx);
                    if (nidx < params.total) {
                        try_enqueue(nidx);
                    }
                }
            }
            atomicSub(&state.activeWorkers, 1u);
            continue;
        }

        // Queue appears empty — exit only when no thread is mid-processing
        // (they could still be adding new items to the queue).
        if (atomicLoad(&state.activeWorkers) == 0u) {
            if (atomicLoad(&state.readHead) >= atomicLoad(&state.writeTail)) {
                break;
            }
        }
    }
}
