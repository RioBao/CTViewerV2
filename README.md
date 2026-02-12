# CTViewer WebGPU

A WebGPU-based industrial CT/medical image viewer built with vanilla TypeScript and Vite. Renders 2D orthogonal slices and 3D Maximum Intensity Projections (MIP) entirely on the GPU.

## Quick Start

```bash
npm install
npm run dev     # Vite dev server with HMR
npm run build   # TypeScript check + production build
```

## Supported Formats

| Format | Extensions | Notes |
|--------|-----------|-------|
| RAW + metadata | `.raw` + `.json` / `.volumeinfo` / `.dat` | Requires paired metadata file with dimensions, data type, spacing |
| DICOM | `.dcm` | Single files, series, multi-frame; auto-sorts by InstanceNumber/SliceLocation |
| NIfTI | `.nii`, `.nii.gz` | NIfTI-1 format, gzip supported |
| TIFF | `.tif`, `.tiff` | Single and multi-page stacks (8/16-bit grayscale) |
| 2D Images | `.png`, `.jpg`, `.bmp`, etc. | Loaded as single-slice volumes via canvas |

## Architecture

```
src/
  main.ts                       # Entry point
  types.ts                      # ViewAxis, VolumeMetadata, VoxelDataType, SliceData
  styles/main.css               # Dark theme with CSS custom properties
  gpu/
    WebGPUContext.ts             # WebGPU device singleton, canvas configuration
    SliceRenderer.ts            # 2D slice renderer (zoom, pan, aspect-correct, crosshairs)
    MIPRenderer.ts              # 3D MIP renderer (orthographic ray marching, Euler camera)
    shaders/
      slice.wgsl                # Aspect-correct slice display + crosshair overlay
      mip.wgsl                  # Orthographic MIP ray marching with gamma correction
  data/
    VolumeData.ts               # 3D volume container, slice extraction, getValue()
  loaders/
    VolumeLoader.ts             # Format dispatcher
    RawLoader.ts                # RAW binary + JSON/volumeinfo/DAT metadata parser
    DicomLoader.ts              # DICOM binary parser (handles non-image files gracefully)
    NiftiLoader.ts              # NIfTI-1 parser with gzip support
    TiffLoader.ts               # TIFF via utif2 (single + multi-page stacking)
    ImageLoader.ts              # 2D images via browser canvas API
  app/
    UIState.ts                  # Typed EventEmitter + viewer state
    FilePicker.ts               # Drag-and-drop + button file selection, file grouping
    ViewerApp.ts                # App orchestrator: rendering, controls, interactions
```

## Features

### 2D Slice Viewing
- Three orthogonal views: XY (Axial), XZ (Coronal), YZ (Sagittal)
- Aspect-correct fitting with zoom and pan
- Scroll through slices with mouse wheel
- Window/level adjustment via histogram handles
- Crosshair overlay with volume coordinate display

### 3D MIP Rendering
- Maximum Intensity Projection via GPU ray marching
- Orthographic projection with Euler angle camera (drag to rotate)
- Configurable quality (200 / 800 / 3200 ray steps)
- Adjustable resolution scaling (0.25x, 0.5x, 1x)
- Gamma correction slider

### Interaction
| Action | Binding |
|--------|---------|
| Pan (2D) | Left-click drag |
| Zoom (2D) | Ctrl + mouse wheel, or `+` / `-` keys |
| Scroll slices | Mouse wheel on 2D viewport |
| Navigate slices | Arrow keys (Left/Right = +/-1, Up/Down = +/-10) |
| Jump to first/last | Home / End |
| Rotate (3D) | Left-click drag on 3D viewport |
| Zoom (3D) | Mouse wheel on 3D viewport |
| Toggle crosshairs | `C` key or Crosshair button |
| Set crosshair | Click on 2D viewport (when crosshairs enabled) |
| Window/level | Drag histogram min/max handles |
| Maximize viewport | Double-click any viewport |
| Restore viewports | Double-click again or Escape |
| Open files | `O` key or Open button |
| Reset view | `R` key or Reset button |
| Fullscreen | `F` key or Fullscreen button |

### Technical Details
- **Zero runtime dependencies** (except utif2 for TIFF decoding)
- **WebGPU-native rendering** — all slice and MIP rendering in WGSL shaders
- **Storage buffers** for slice data (not textures) — supports any numeric data type
- **Adapter-aware buffer limits** — requests maximum GPU buffer size for large volumes
- **DPR-aware** — 2D canvases render at device pixel ratio for sharp display
- **Dark theme** with CSS custom properties
- **Streaming architecture for 4GB+ files** — never loads full data into memory:
  - Three loading modes: Standard (<200MB full load), Hybrid (200MB–2GB: preview + background full load), Streaming (>2GB: on-demand slices only)
  - XY slices read contiguously from file (~4MB, <50ms) with LRU cache (100 slices) and concurrent prefetch (cap=4)
  - XZ/YZ slices use slab-based progressive loading (center-outward, ~50MB slabs, zero-copy typed array views for row extraction)
  - Cancellation support — aborts in-progress loads when user scrolls away
  - Low-res 4x-downsampled preview for instant display and MIP rendering

## Conventions

- Vanilla TypeScript, no UI framework
- Strict mode (`noUnusedLocals`, `noUnusedParameters`)
- ES2022 target
- WGSL shaders imported as raw strings via Vite
- EventEmitter pattern for state management
