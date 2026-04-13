# RFC API Implementation Status

Date: 2026-03-13
Scope: `backends/qualcomm/utils/fx_viewer/templates/*`
Reference: `backends/qualcomm/utils/fx_viewer/RFC_FX_VIEWER_API_INTERFACE.md`

## Summary

Most RFC APIs are implemented in the current JS runtime.
Remaining gaps are minor and documented below.

## Implemented

1. Construction and presets
- `FXGraphViewer.create(config)`
- Presets: `split`, `compact`, `headless`, `custom`
- Slot precedence and layout merge behavior

2. Canonical state API
- `getState`
- `setState`
- `replaceState` (state replacement with camera/search handling)
- `batch`

3. Convenience APIs
- `setTheme`, `setLayers`, `setColorBy`
- `selectNode`, `clearSelection`, `search`, `zoomToFit`, `panToNode`, `animateToNode`
- `setUIVisibility`, `setLayout`
- `enterFullscreen`, `exitFullscreen`, `destroy`

4. Runtime layer mutation APIs
- `upsertLayer`, `removeLayer`, `patchLayerNodes`, `setLayerLabel`, `setColorRule`

5. Events
- `statechange`, `selectionchange`, `themechange`, `layoutchange`, `error`
- `on`/`off` subscription model

6. Compare API
- `FXGraphCompare.create`
- `setColumns` (applies to optional compare container)
- `setCompact`, `setSync`, `destroy`

7. UI synchronization contract
- External state updates reflected in theme/layers/colorBy controls
- `syncControlsFromState()` in `UIManager`

8. Fullscreen taskbar support
- Optional taskbar fullscreen button via `layout.fullscreen.button` / `ui.controls.fullscreenButton`

## Partial / Follow-up

1. Strict state schema validation
- RFC describes validation-rich state store; current implementation uses pragmatic checks and coercions.

2. Theme registration depth
- `registerTheme` works; deeper token validation and compatibility checks are not yet strict.

3. Compare camera/theme/layer sync
- Compare selection sync is implemented.
- Other sync dimensions are modeled in config but not fully propagated yet.
