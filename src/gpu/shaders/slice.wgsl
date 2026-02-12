// Slice rendering shader
// Renders a 2D grayscale slice with zoom, pan, aspect-correct fitting, and crosshair overlay

struct Uniforms {
    windowMin: f32,
    windowMax: f32,
    sliceWidth: f32,
    sliceHeight: f32,
    canvasWidth: f32,
    canvasHeight: f32,
    zoom: f32,
    panX: f32,
    panY: f32,
    crosshairX: f32,     // in slice pixel coords
    crosshairY: f32,     // in slice pixel coords
    crosshairEnabled: f32, // > 0.5 = on
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> sliceData: array<f32>;

// Fullscreen triangle: 3 vertices, no vertex buffer needed
@vertex
fn vs(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vertexIndex & 1u)) * 4.0 - 1.0;
    let y = f32(i32(vertexIndex >> 1u)) * 4.0 - 1.0;
    out.position = vec4f(x, y, 0.0, 1.0);
    // UV: [0,1] range, flip Y so top-left is (0,0)
    out.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4f {
    let w = u.sliceWidth;
    let h = u.sliceHeight;

    // Canvas pixel position centered at origin
    let canvasPos = (in.uv - 0.5) * vec2f(u.canvasWidth, u.canvasHeight);

    // Fit scale: canvas pixels per slice pixel (aspect-correct contain)
    let fitScale = min(u.canvasWidth / w, u.canvasHeight / h);

    // Map canvas position to slice pixel coords with zoom and pan
    let slicePos = canvasPos / (fitScale * u.zoom) - vec2f(u.panX, u.panY) + vec2f(w, h) * 0.5;

    // Crosshair overlay (draw on top regardless of bounds)
    if (u.crosshairEnabled > 0.5) {
        let chSlice = vec2f(u.crosshairX, u.crosshairY);
        // Convert crosshair from slice to canvas space
        let chCanvas = (chSlice - vec2f(w, h) * 0.5 + vec2f(u.panX, u.panY)) * fitScale * u.zoom;
        // Distance in canvas pixels
        let dist = abs(canvasPos - chCanvas);
        if (dist.x < 0.75 || dist.y < 0.75) {
            return vec4f(1.0, 1.0, 0.0, 1.0); // yellow crosshair
        }
    }

    // Bounds check
    if (slicePos.x < 0.0 || slicePos.x >= w || slicePos.y < 0.0 || slicePos.y >= h) {
        return vec4f(0.05, 0.05, 0.05, 1.0); // dark background outside slice
    }

    // Bilinear interpolation for smooth zoomed display
    let px = clamp(slicePos.x, 0.0, w - 1.001);
    let py = clamp(slicePos.y, 0.0, h - 1.001);

    let x0 = u32(floor(px));
    let y0 = u32(floor(py));
    let x1 = min(x0 + 1u, u32(w) - 1u);
    let y1 = min(y0 + 1u, u32(h) - 1u);

    let fx = px - floor(px);
    let fy = py - floor(py);

    let width = u32(w);
    let v00 = sliceData[y0 * width + x0];
    let v10 = sliceData[y0 * width + x1];
    let v01 = sliceData[y1 * width + x0];
    let v11 = sliceData[y1 * width + x1];

    let v0 = mix(v00, v10, fx);
    let v1 = mix(v01, v11, fx);
    let value = mix(v0, v1, fy);

    // Window/level mapping
    let range = u.windowMax - u.windowMin;
    let normalized = clamp((value - u.windowMin) / max(range, 0.001), 0.0, 1.0);

    return vec4f(normalized, normalized, normalized, 1.0);
}
