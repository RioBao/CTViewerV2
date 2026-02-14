// Volume ray marching shader
// Supports MIP, MinIP, Average, and Compositing (DVR) modes
// Hardware-filtered 3D texture, transfer function, Blinn-Phong lighting
// Ray-AABB intersection, brick-map empty space skipping, adaptive stepping

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
    // Rendering
    frameCount: f32,
    renderMode: f32,
    opacityScale: f32,
    // Lighting
    ambient: f32,
    diffuse: f32,
    specular: f32,
    shininess: f32,
    lightingEnabled: f32,
    // Brick map
    brickGridX: f32,
    brickGridY: f32,
    brickGridZ: f32,
    maskOverlayEnabled: f32,
    maskOverlayOpacity: f32,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var volumeTex: texture_3d<f32>;
@group(0) @binding(2) var volumeSampler: sampler;
@group(0) @binding(3) var tfTexture: texture_2d<f32>;
@group(0) @binding(4) var tfSampler: sampler;
@group(0) @binding(5) var brickTex: texture_3d<f32>;
@group(0) @binding(6) var brickSampler: sampler;
@group(0) @binding(7) var labelTex: texture_3d<u32>;
@group(0) @binding(8) var maskPaletteTex: texture_2d<f32>;

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

// Sample volume using hardware-filtered 3D texture
fn sampleVolume(voxPos: vec3f) -> f32 {
    let dims = vec3f(u.dimX, u.dimY, u.dimZ);
    let uvw = (voxPos + 0.5) / dims;
    return textureSampleLevel(volumeTex, volumeSampler, uvw, 0.0).r;
}

// Hash for jittered ray start to eliminate banding
fn hash(p: vec2f) -> f32 {
    return fract(sin(dot(p, vec2f(12.9898, 78.233))) * 43758.5453);
}

// Transfer function lookup
fn lookupTF(intensity: f32) -> vec4f {
    let range = u.displayMax - u.displayMin;
    let t = clamp((intensity - u.displayMin) / max(range, 0.001), 0.0, 1.0);
    return textureSampleLevel(tfTexture, tfSampler, vec2f(t, 0.5), 0.0);
}

// Central-difference gradient (6 texture samples)
fn computeGradient(voxPos: vec3f) -> vec3f {
    return vec3f(
        sampleVolume(voxPos + vec3f(1.0, 0.0, 0.0)) - sampleVolume(voxPos - vec3f(1.0, 0.0, 0.0)),
        sampleVolume(voxPos + vec3f(0.0, 1.0, 0.0)) - sampleVolume(voxPos - vec3f(0.0, 1.0, 0.0)),
        sampleVolume(voxPos + vec3f(0.0, 0.0, 1.0)) - sampleVolume(voxPos - vec3f(0.0, 0.0, 1.0))
    ) * 0.5;
}

// Blinn-Phong shading with headlight (light = camera view direction)
fn blinnPhong(normal: vec3f, viewDir: vec3f, baseColor: vec3f) -> vec3f {
    let n = normalize(normal);
    let h = normalize(viewDir + viewDir); // headlight: L = V
    let diff = u.diffuse * max(dot(n, viewDir), 0.0);
    let spec = u.specular * pow(max(dot(n, h), 0.0), u.shininess);
    return u.ambient * baseColor + diff * baseColor + spec * vec3f(1.0);
}

// Build turntable camera matrix with Z-up
fn rotationMatrix(az: f32, el: f32, ro: f32) -> mat3x3f {
    let sa = sin(az); let ca = cos(az);
    let se = sin(el); let ce = cos(el);
    let sr = sin(ro); let cr = cos(ro);

    let right0 = vec3f(-sa, ca, 0.0);
    let up0 = vec3f(-ca * se, -sa * se, ce);
    let fwd0 = vec3f(-ce * ca, -ce * sa, -se);

    let right = right0 * cr + up0 * sr;
    let up = -right0 * sr + up0 * cr;

    return mat3x3f(right, up, fwd0);
}

// Ray-AABB intersection: returns (tNear, tFar)
fn intersectAABB(origin: vec3f, dir: vec3f, boxMin: vec3f, boxMax: vec3f) -> vec2f {
    let invDir = 1.0 / dir;
    let t1 = (boxMin - origin) * invDir;
    let t2 = (boxMax - origin) * invDir;
    let tMin = min(t1, t2);
    let tMax = max(t1, t2);
    let tNear = max(tMin.x, max(tMin.y, tMin.z));
    let tFar  = min(tMax.x, min(tMax.y, tMax.z));
    return vec2f(max(tNear, 0.0), tFar);
}

// Brick map lookup: returns (min, max) intensity for the 8³ brick containing voxPos
fn lookupBrick(voxPos: vec3f) -> vec2f {
    let brickUVW = (floor(voxPos / 8.0) + 0.5) / vec3f(u.brickGridX, u.brickGridY, u.brickGridZ);
    return textureSampleLevel(brickTex, brickSampler, brickUVW, 0.0).rg;
}

fn sampleLabel(voxPos: vec3f) -> u32 {
    let maxCoord = vec3i(
        max(0, i32(u.dimX) - 1),
        max(0, i32(u.dimY) - 1),
        max(0, i32(u.dimZ) - 1),
    );
    let coord = clamp(vec3i(floor(voxPos)), vec3i(0), maxCoord);
    return textureLoad(labelTex, coord, 0).r;
}

fn lookupMaskPalette(classId: u32) -> vec4f {
    let index = clamp(i32(classId), 0, 255);
    return textureLoad(maskPaletteTex, vec2i(index, 0), 0);
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

    // Starting point: offset by screen position, far behind volume center
    let rayOrigin = screenPos.x * rayRight + screenPos.y * rayUp - 1.5 * rayDir;

    // Ray-AABB intersection to clip march to volume bounds
    let boxMin = -normSize * 0.5;
    let boxMax =  normSize * 0.5;
    let tRange = intersectAABB(rayOrigin, rayDir, boxMin, boxMax);

    if (tRange.x >= tRange.y) {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }

    // Convert ray to voxel space
    let toVoxel = dims / normSize;
    let center = dims * 0.5;

    let numSteps = u32(u.numSteps);
    let baseStep = (tRange.y - tRange.x) / f32(numSteps);

    // Brick size in normalized space (one brick = 8 voxels along largest axis)
    let brickNormSize = 8.0 / max(toVoxel.x, max(toVoxel.y, toVoxel.z));

    // Jittered ray start to eliminate banding
    let jitter = hash(in.position.xy + vec2f(u.frameCount * 0.7919, 0.0));
    let tStart = tRange.x + jitter * baseStep;

    let mode = u32(u.renderMode);

    // Mode-specific accumulators
    var maxVal: f32 = -1e20;
    var minVal: f32 = 1e20;
    var sumVal: f32 = 0.0;
    var sampleCount: f32 = 0.0;
    var accumColor = vec4f(0.0, 0.0, 0.0, 0.0);
    var overlayAccum = vec4f(0.0, 0.0, 0.0, 0.0);
    var hitVolume = false;

    var t = tStart;

    for (var i = 0u; i < numSteps; i++) {
        if (t > tRange.y) { break; }

        let pos = rayOrigin + rayDir * t;

        // Convert normalized position to voxel coordinates
        let voxPos = pos * toVoxel + center;

        // Brick map skip: check if brick is empty
        let brick = lookupBrick(voxPos);
        if (brick.y < u.displayMin) {
            // Empty brick — skip ahead by one brick width
            t += brickNormSize;
            continue;
        }

        let val = sampleVolume(voxPos);
        hitVolume = true;

        // Adaptive step: fine near surfaces (high brick range), coarse in uniform
        let brickRange = brick.y - brick.x;
        let volRange = u.displayMax - u.displayMin;
        let surfaceRatio = clamp(brickRange / max(volRange * 0.1, 0.001), 0.0, 1.0);
        let adaptiveStep = mix(baseStep * 2.0, baseStep * 0.5, surfaceRatio);

        if (u.maskOverlayEnabled > 0.5) {
            let classId = sampleLabel(voxPos);
            if (classId > 0u) {
                let palette = lookupMaskPalette(classId);
                let overlayAlpha = clamp(palette.a * u.maskOverlayOpacity * adaptiveStep * 80.0, 0.0, 1.0);
                if (overlayAlpha > 0.0001) {
                    let srcOverlay = vec4f(palette.rgb * overlayAlpha, overlayAlpha);
                    overlayAccum += srcOverlay * (1.0 - overlayAccum.a);
                }
            }
        }

        switch mode {
            case 0u: { // MIP
                maxVal = max(maxVal, val);
            }
            case 1u: { // MinIP
                minVal = min(minVal, val);
            }
            case 2u: { // Average
                sumVal += val;
                sampleCount += 1.0;
            }
            case 3u: { // Compositing (front-to-back)
                let tf = lookupTF(val);
                // Scale alpha by adaptiveStep/baseStep to compensate for variable step size
                let alpha = tf.a * u.opacityScale * adaptiveStep * 100.0;
                var color = tf.rgb;

                // Apply lighting if enabled and sample is visible
                if (u.lightingEnabled > 0.5 && alpha > 0.001) {
                    let grad = computeGradient(voxPos);
                    let spacingInv = vec3f(1.0 / u.spacingX, 1.0 / u.spacingY, 1.0 / u.spacingZ);
                    let worldGrad = grad * spacingInv;
                    if (length(worldGrad) > 0.01) {
                        color = blinnPhong(worldGrad, -rayDir, color);
                    }
                }

                let src = vec4f(color * alpha, alpha);
                accumColor += src * (1.0 - accumColor.a);

                // Early ray termination
                if (accumColor.a >= 0.95) {
                    break;
                }
            }
            default: {
                maxVal = max(maxVal, val);
            }
        }

        t += adaptiveStep;
    }

    if (!hitVolume) {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }

    // Mode 3 (Compositing): output accumulated color directly
    if (mode == 3u) {
        let blended = mix(accumColor.rgb, overlayAccum.rgb, overlayAccum.a);
        return vec4f(blended, 1.0);
    }

    // For projection modes, determine the result value
    var resultVal: f32;
    switch mode {
        case 0u: { resultVal = maxVal; }
        case 1u: { resultVal = minVal; }
        case 2u: {
            resultVal = select(0.0, sumVal / sampleCount, sampleCount > 0.0);
        }
        default: { resultVal = maxVal; }
    }

    // Window/level mapping
    let range = u.displayMax - u.displayMin;
    let normalized = clamp((resultVal - u.displayMin) / max(range, 0.001), 0.0, 1.0);

    // Gamma correction
    let corrected = pow(normalized, u.gamma);

    let base = vec3f(corrected, corrected, corrected);
    let blended = mix(base, overlayAccum.rgb, overlayAccum.a);
    return vec4f(blended, 1.0);
}
