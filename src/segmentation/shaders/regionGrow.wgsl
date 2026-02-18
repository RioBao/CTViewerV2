struct Params {
    width: u32,
    height: u32,
    total: u32,
    tolerance: f32,
    seedValue: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// Monotonic frontier: frontier[0..nextEnd] holds all selected pixel indices.
// The current BFS level occupies frontier[levelStart..levelEnd].
// nextEnd grows as new pixels are discovered; levelStart/levelEnd advance each step.
struct BfsState {
    levelStart: u32,
    levelEnd: u32,
    nextEnd: atomic<u32>,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>            values:      array<f32>;
@group(0) @binding(1) var<storage, read_write>      visited:     array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write>      frontier:    array<u32>;
@group(0) @binding(3) var<storage, read_write>      state:       BfsState;
@group(0) @binding(4) var<storage, read_write>      indirectArgs: array<u32>; // [x, y, z]
@group(0) @binding(5) var<storage, read>            params:      Params;

const WORKGROUP_SIZE: u32 = 256u;

// Process one BFS level: each thread handles one pixel from frontier[levelStart..levelEnd],
// examining its 8 neighbours and appending unvisited, in-tolerance ones to the frontier.
@compute @workgroup_size(WORKGROUP_SIZE)
fn bfs_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let levelSize = state.levelEnd - state.levelStart;
    if (i >= levelSize) { return; }

    let idx = frontier[state.levelStart + i];
    let x = idx % params.width;
    let y = idx / params.width;

    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            if (dx == 0 && dy == 0) { continue; }
            let nx = i32(x) + dx;
            let ny = i32(y) + dy;
            if (nx < 0 || ny < 0 || nx >= i32(params.width) || ny >= i32(params.height)) { continue; }
            let nidx = u32(ny) * params.width + u32(nx);

            // Atomically claim the pixel (0 → 1). Only the thread that wins the CAS proceeds.
            let prev = atomicCompareExchangeWeak(&visited[nidx], 0u, 1u);
            if (prev.old_value == 0u) {
                let delta = abs(values[nidx] - params.seedValue);
                if (delta <= params.tolerance) {
                    let slot = atomicAdd(&state.nextEnd, 1u);
                    frontier[slot] = nidx;
                }
            }
        }
    }
}

// Advance the BFS window and write the workgroup count for the next bfs_step
// into indirectArgs so the GPU can drive the next dispatch without a CPU round-trip.
@compute @workgroup_size(1)
fn prepare_next() {
    let next = atomicLoad(&state.nextEnd);
    state.levelStart = state.levelEnd;
    state.levelEnd = next;

    let levelSize = next - state.levelStart;
    // Always write at least 1 so the indirect buffer stays valid; bfs_step returns
    // early when i >= levelSize, making a 0-level dispatch a harmless no-op.
    let wgCount = max(1u, (levelSize + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE);
    indirectArgs[0] = wgCount;
    indirectArgs[1] = 1u;
    indirectArgs[2] = 1u;
}
