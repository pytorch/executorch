# FX Viewer API Harness Testcases

This document tracks testcase intent and API coverage for the unified harness.

## Harness Outputs

1. `fx_viewer_api_test_harness_portable.html`
2. `fx_viewer_api_test_harness_qualcomm.html`

Portable harness requires no Qualcomm SDK.
Qualcomm harness requires QAIRT/QNN environment.

## Testcases

1. `topology_split`
- Goal: baseline split viewer using structural extensions.
- APIs: `create`, `init`, `setLayers`, `setColorBy`.

2. `headless_slots`
- Goal: host-controlled layout with external slots and dynamic recoloring.
- APIs: `create` with `mount.slots`, `patchLayerNodes`, `setColorBy`.

3. `accuracy_dynamic`
- Goal: real per-layer accuracy debug controls (threshold/theme/focus).
- APIs: `setTheme`, `selectNode`, `patchLayerNodes`, `setColorBy`.

4. `fullscreen_toolbar`
- Goal: demonstrate fullscreen integration in UI + direct API calls.
- APIs: `layout.fullscreen.button`, `enterFullscreen`, `exitFullscreen`.

5. `compare_sync`
- Goal: compare orchestration with sync toggles.
- APIs: `FXGraphCompare.create`, `setSync`, `setCompact`.

6. `qualcomm_metadata` (Qualcomm harness only)
- Goal: expose Qualcomm PTQ metadata next to real graph payload.
- APIs: `create`, metadata-driven host composition.
