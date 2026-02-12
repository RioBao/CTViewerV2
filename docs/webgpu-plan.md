# WebGPU Viewer Plan (Ambitious Fork)

## Goal
Build a new WebGPU-based viewer that keeps fast low-res loading while adding high-quality 3D rendering, segmentation, and AI inference (local + HuggingFace). The current WebGL2 viewer remains stable and unchanged.

## Target Platforms
- Chrome + Edge (WebGPU)
- Desktop focus first

## Guiding Principles
- First pixel fast: show low-res preview quickly.
- Progressive refinement: stream higher detail without UI stalls.
- GPU-first: compute and rendering on WebGPU when possible.
- Separation of concerns: clean module boundaries for volume, render, segmentation, AI.
- Binary now, multi-class later: design data structures to scale.

## Phases

### Phase 0: Scaffold
- New repo `CTViewer-WebGPU`.
- WebGPU init + graceful fallback if unavailable.
- Basic UI shell + file picker.

### Phase 1: Fast Loading
- Parse metadata + raw stream.
- Generate low-res pyramid (CPU first, GPU later).
- Upload low-res 3D texture for immediate preview.
- Progressive upload of full-res chunks.
- Streaming for on-the-fly downsampling of blocks currently outside the users view 
- UI shows current resolution + load progress.
- Coordinate Transforms, scale voxels correctly using a world matrix

### Phase 2: Rendering
- Slice views (xy/xz/yz) on GPU.
- Volume raymarch shader with transfer function.
- Quality slider: sample step / resolution.
- Basic lighting for 3D volume.

### Phase 3: Segmentation (Binary)
- Mask buffer (uint8) + overlay. Dirty-Rect/Sub-region update strategy for mask synchronization.
- Use bit-masks
- Tools: threshold, brush, region grow.
- Mask save/export.

### Phase 4: AI Integration
- Local: ONNX Runtime Web (WebGPU backend, WASM fallback).
- Remote: HuggingFace endpoint (slice-based first).
- Unified inference interface for model selection.
- UI toggle: Local vs Remote.

### Phase 5: Multi-class Segmentation
- Mask format supports class IDs.
- UI class palette, per-class overlay toggles.

### Phase 6: 3D Inference
- ROI extraction for 3D models.
- Optional full-volume upload (small volumes only).

## Deliverables by Phase
- P1: Low-res preview + progressive refinement
- P2: WebGPU raymarching renderer + slice views
- P3: Binary segmentation tools
- P4: Local + HF inference
- P5: Multi-class support
- P6: 3D inference pipeline

## Risks and Mitigations
- Large volumes (1-5GB): must stream and downsample early.
- Browser limitations: keep graceful fallback and memory safety checks.
- Model sizes: ship small default models, allow user to load custom.

## Success Metrics
- Low-res preview under 2-3 seconds for typical data.
- Smooth interaction at >30 FPS during navigation.
- Segmentation tools responsive (<100ms).
