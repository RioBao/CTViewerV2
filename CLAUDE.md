# CTViewer WebGPU

WebGPU-based Industrial CT Viewer — modern replacement for the vanilla JS/WebGL2 viewer at `D:\Programming\Viewer`.

## Quick Start

```bash
npm install
npm run dev     # Vite dev server with HMR
npm run build   # TypeScript check + production build
```

## Architecture

```
src/
  main.ts                       # Entry point: imports CSS, boots ViewerApp
  types.ts                      # Shared types: ViewAxis, VolumeMetadata, etc.
  styles/main.css               # Dark theme CSS with custom properties
  gpu/
    WebGPUContext.ts             # WebGPU device singleton, canvas configuration
    SliceRenderer.ts            # 2D slice renderer (storage buffer + window/level)
    MIPRenderer.ts              # 3D MIP renderer (ray marching, Euler camera)
    shaders/slice.wgsl          # Fullscreen triangle + grayscale fragment shader
    shaders/mip.wgsl            # Orthographic MIP ray marching shader
  data/
    VolumeData.ts               # 3D volume container, slice extraction, min/max
  loaders/
    VolumeLoader.ts             # Dispatch FileGroup → appropriate loader
    RawLoader.ts                # RAW + JSON/volumeinfo/DAT metadata
    DicomLoader.ts              # DICOM binary parser, series/multi-frame
    NiftiLoader.ts              # NIfTI-1 (.nii, .nii.gz)
    TiffLoader.ts               # Single + multi-page TIFF via utif2
    ImageLoader.ts              # 2D images (PNG, JPG, etc.) via canvas
  app/
    UIState.ts                  # Typed EventEmitter + viewer state container
    FilePicker.ts               # Drag-and-drop & button file selection, file grouping
    ViewerApp.ts                # App orchestrator: init, controls, resize
```

## Conventions

- **Vanilla TypeScript** — no UI framework, no runtime deps
- **Strict mode** — `noUnusedLocals`, `noUnusedParameters`
- **ES2022 target** — top-level await, private fields OK
- **CSS custom properties** for theming (`--bg-primary`, `--accent`, etc.)
- **EventEmitter pattern** for state changes (UIState.on/emit)
- Shader files use `.wgsl` extension (imported as assets via Vite)

## Phase Roadmap

- **Phase 0** (done): Scaffold — build tooling, UI shell, WebGPU init, file picker
- **Phase 1** (done): Volume loading — RAW, DICOM, NIfTI, TIFF, 2D image parsers
- **Phase 2** (done): 2D slice rendering — WebGPU shader pipeline, scroll navigation
- **Phase 3** (done): 3D MIP rendering — ray marching in fragment shader
- **Phase 4** (done): Interaction — zoom, pan, window/level, crosshairs, histogram, keyboard shortcuts, double-click maximize
- **Phase 5** (done): Polish — progressive loading, ROI selection, histogram DPI, error states
