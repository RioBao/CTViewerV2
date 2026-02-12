# WebGPU Viewer Technical Spec (Fork)

## 1. Scope
Build a WebGPU-based CT viewer with:
- Fast low-res loading + progressive refinement.
- High-quality 3D rendering (raymarch).
- Segmentation (binary first, multi-class later).
- AI inference (local ONNX + remote HuggingFace).

This spec targets a new repo (`CTViewer-WebGPU`). The existing WebGL2 viewer stays stable.

## 2. Browser Targets
- Chrome and Edge (WebGPU enabled).
- Fallback behavior: display error + suggest using the WebGL viewer.

## 3. Data & Loading

### 3.1 Metadata Formats (initial)
- `.json` / `.volumeinfo` / `.dat` for RAW volumes.
- DICOM/NIfTI later (post-core).

### 3.2 Loading Strategy
- Stage 1: parse metadata, start streaming raw.
- Stage 2: create low-res pyramid (downsample to fit memory).
- Stage 3: upload low-res to GPU for immediate display.
- Stage 4: progressively load full-res chunks into GPU storage.

### 3.3 Memory Budget
- Soft cap: ~512MB GPU for volume data.
- Downsample when exceeding cap.

## 4. Rendering

### 4.1 Slice Views
- Compute shader or render pipeline for extracting slices.
- Shared LUT for window/level or transfer function.

### 4.2 3D Raymarching
- Volume texture + transfer function.
- Adjustable step size (quality slider).
- Optional lighting (Phong or gradient-based).

## 5. Segmentation

### 5.1 Binary Mask (Phase 1)
- Mask volume: uint8 (0/1).
- Overlay rendering: GPU blending.
- Tools: threshold, brush, region grow.

### 5.2 Multi-Class (Phase 2)
- Mask volume: uint8 or uint16 class IDs.
- Class palette stored in UI state.
- Overlay per-class toggles.

## 6. AI Integration

### 6.1 Local (ONNX Runtime Web)
- Prefer WebGPU backend.
- WASM fallback.
- Models are 2D initially (per slice).

### 6.2 Remote (HuggingFace)
- Default input: slice PNG or float array.
- Optional full volume upload for small datasets.
- Simple REST client:
  - `POST /infer` with metadata
  - `GET /status` (if async)

### 6.3 Unified Interface
`InferenceClient` with:
- `runSlice(input, options)`
- `runVolumeROI(volume, bbox, options)`
- `getCapabilities()`

## 7. File Structure (Proposed)
```
CTViewer-WebGPU/
  index.html
  package.json
  src/
    main.ts
    app/
      ViewerApp.ts
      UIState.ts
    volume/
      VolumeSource.ts
      VolumePyramid.ts
      VolumeCache.ts
    gpu/
      WebGPUDevice.ts
      Pipelines.ts
      Shaders.wgsl
    render/
      SliceRenderer.ts
      RaymarchRenderer.ts
      TransferFunction.ts
    segmentation/
      MaskVolume.ts
      MaskRenderer.ts
      Tools.ts
    ai/
      InferenceClient.ts
      LocalOnnxRunner.ts
      RemoteHfRunner.ts
  public/
    models/
  docs/
    webgpu-plan.md
    webgpu-spec.md
```

## 8. Minimal Bootstrap Code

### `index.html`
```html
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>CTViewer WebGPU</title>
  </head>
  <body>
    <canvas id="gpuCanvas" width="1024" height="768"></canvas>
    <script type="module" src="/src/main.ts"></script>
  </body>
  </html>
```

### `src/main.ts`
```ts
async function initWebGPU() {
  if (!navigator.gpu) {
    throw new Error("WebGPU not supported.");
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No GPU adapter.");
  const device = await adapter.requestDevice();

  const canvas = document.getElementById("gpuCanvas") as HTMLCanvasElement;
  const context = canvas.getContext("webgpu") as GPUCanvasContext;

  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format,
    alphaMode: "opaque"
  });

  // Clear to a dark gray as a minimal render loop.
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      loadOp: "clear",
      storeOp: "store",
      clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1 }
    }]
  });
  pass.end();
  device.queue.submit([encoder.finish()]);
}

initWebGPU().catch(console.error);
```

## 9. Next Steps
- Scaffold repo with this structure.
- Implement low-res volume upload pipeline.
- Add raymarch shader and slice renderer.
- Add binary mask pipeline.
- Integrate ONNX Runtime Web + HF client.
