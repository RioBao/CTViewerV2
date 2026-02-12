// Maximum Intensity Projection (MIP) ray marching shader
// Orthographic projection through a 3D volume stored as a flat storage buffer

struct Uniforms {
    // Camera (radians)
    azimuth: f32,
    elevation: f32,
    roll: f32,
    distance: f32,
    // Pan (normalized screen coords)
    panX: f32,
    panY: f32,
    // Display
    displayMin: f32,
    displayMax: f32,
    gamma: f32,
    // Volume
    dimX: f32,
    dimY: f32,
    dimZ: f32,
    spacingX: f32,
    spacingY: f32,
    spacingZ: f32,
    numSteps: f32,
    // Canvas
    canvasWidth: f32,
    canvasHeight: f32,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> volume: array<f32>;

// Fullscreen triangle
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vi & 1u)) * 4.0 - 1.0;
    let y = f32(i32(vi >> 1u)) * 4.0 - 1.0;
    out.position = vec4f(x, y, 0.0, 1.0);
    out.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// Sample volume with trilinear interpolation for smooth rendering
fn sampleVolume(pos: vec3f) -> f32 {
    // Clamp to valid range
    let px = clamp(pos.x, 0.0, u.dimX - 1.001);
    let py = clamp(pos.y, 0.0, u.dimY - 1.001);
    let pz = clamp(pos.z, 0.0, u.dimZ - 1.001);

    // Get integer coordinates of 8 surrounding voxels
    let x0 = u32(floor(px));
    let y0 = u32(floor(py));
    let z0 = u32(floor(pz));
    let x1 = min(x0 + 1u, u32(u.dimX) - 1u);
    let y1 = min(y0 + 1u, u32(u.dimY) - 1u);
    let z1 = min(z0 + 1u, u32(u.dimZ) - 1u);

    // Fractional parts for interpolation
    let fx = px - floor(px);
    let fy = py - floor(py);
    let fz = pz - floor(pz);

    // Sample 8 corners of the cube
    let nx = u32(u.dimX);
    let nxy = u32(u.dimX) * u32(u.dimY);

    let v000 = volume[x0 + y0 * nx + z0 * nxy];
    let v100 = volume[x1 + y0 * nx + z0 * nxy];
    let v010 = volume[x0 + y1 * nx + z0 * nxy];
    let v110 = volume[x1 + y1 * nx + z0 * nxy];
    let v001 = volume[x0 + y0 * nx + z1 * nxy];
    let v101 = volume[x1 + y0 * nx + z1 * nxy];
    let v011 = volume[x0 + y1 * nx + z1 * nxy];
    let v111 = volume[x1 + y1 * nx + z1 * nxy];

    // Trilinear interpolation
    let v00 = mix(v000, v100, fx);
    let v01 = mix(v001, v101, fx);
    let v10 = mix(v010, v110, fx);
    let v11 = mix(v011, v111, fx);

    let v0 = mix(v00, v10, fy);
    let v1 = mix(v01, v11, fy);

    return mix(v0, v1, fz);
}

// Build turntable camera matrix with Z-up
// az: rotation around world Z axis (turntable spin)
// el: elevation angle above XY plane (0 = side, pi/2 = top-down)
// ro: roll around view direction
fn rotationMatrix(az: f32, el: f32, ro: f32) -> mat3x3f {
    let sa = sin(az); let ca = cos(az);
    let se = sin(el); let ce = cos(el);
    let sr = sin(ro); let cr = cos(ro);

    // Turntable base vectors (Z-up, looking toward center)
    let right0 = vec3f(-sa, ca, 0.0);
    let up0 = vec3f(-ca * se, -sa * se, ce);
    let fwd0 = vec3f(-ce * ca, -ce * sa, -se);

    // Apply roll around view direction
    let right = right0 * cr + up0 * sr;
    let up = -right0 * sr + up0 * cr;

    return mat3x3f(right, up, fwd0);
}

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4f {
    let dims = vec3f(u.dimX, u.dimY, u.dimZ);
    let spacing = vec3f(u.spacingX, u.spacingY, u.spacingZ);

    // Volume physical size (in world units)
    let physSize = dims * spacing;
    let maxPhys = max(physSize.x, max(physSize.y, physSize.z));

    // Normalize to [-0.5, 0.5] range (largest dim = 1.0)
    let normSize = physSize / maxPhys;

    // Camera rotation matrix
    let rot = rotationMatrix(u.azimuth, u.elevation, u.roll);

    // Ray origin in normalized view space (orthographic, aspect-corrected)
    let aspect = u.canvasWidth / max(u.canvasHeight, 1.0);
    let screenPos = vec2f(
        (in.uv.x - 0.5) * aspect / u.distance + u.panX,
        (in.uv.y - 0.5) / u.distance + u.panY,
    );

    // Ray direction is the forward vector of the camera
    let rayDir = rot[2]; // forward column
    let rayRight = rot[0];
    let rayUp = rot[1];

    // Starting point: far behind the volume center, offset by screen position
    let rayOrigin = screenPos.x * rayRight + screenPos.y * rayUp - 1.5 * rayDir;

    // Convert ray to voxel space
    // Normalized space [-0.5, 0.5] → voxel space [0, dim]
    let toVoxel = dims / normSize;
    let center = dims * 0.5;

    let numSteps = u32(u.numSteps);
    // Step through 3 units of diagonal (enough to traverse the volume)
    let totalDist = 3.0;
    let stepSize = totalDist / f32(numSteps);

    var maxVal: f32 = -1e20;
    var hitVolume = false;

    for (var i = 0u; i < numSteps; i++) {
        let t = f32(i) * stepSize;
        let pos = rayOrigin + rayDir * t;

        // Convert normalized position to voxel coordinates
        let voxPos = pos * toVoxel + center;

        // Check bounds
        if (voxPos.x >= 0.0 && voxPos.x < dims.x &&
            voxPos.y >= 0.0 && voxPos.y < dims.y &&
            voxPos.z >= 0.0 && voxPos.z < dims.z) {
            let val = sampleVolume(voxPos);
            maxVal = max(maxVal, val);
            hitVolume = true;
        }
    }

    if (!hitVolume) {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }

    // Window/level mapping
    let range = u.displayMax - u.displayMin;
    let normalized = clamp((maxVal - u.displayMin) / max(range, 0.001), 0.0, 1.0);

    // Gamma correction
    let corrected = pow(normalized, u.gamma);

    return vec4f(corrected, corrected, corrected, 1.0);
}
