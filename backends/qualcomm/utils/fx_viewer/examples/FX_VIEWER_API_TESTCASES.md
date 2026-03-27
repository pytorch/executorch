# FX Viewer API Harness: Tutorial Testcases

This document is a learning guide for the unified harness.
Read top-to-bottom and run cases in order.

## Harness Outputs

1. `fx_viewer_api_test_harness_portable.html`
2. `fx_viewer_api_test_harness_qualcomm.html`

Portable harness requires no Qualcomm SDK.
Qualcomm harness requires QAIRT/QNN environment.

## Learning Order

### Level 1: API Fundamentals

1. `js_01_create_init_destroy`
- Purpose: learn viewer lifecycle.
- Target APIs: `FXGraphViewer.create`, `init`, `destroy`, `getState`.
- What to try:
1. Click `Create + Init`.
2. Click `Destroy`.
3. Repeat and observe state panel.

2. `js_02_state_theme`
- Purpose: understand state-driven controls.
- Target APIs: `getState`, `setState`, `setTheme`.
- What to try:
1. Switch light/dark theme.
2. Toggle highlight mode.
3. Watch `getState()` snapshot update.

3. `js_03_selection_camera`
- Purpose: learn navigation semantics.
- Target APIs: `selectNode`, `animateToNode`, `panToNode`, `zoomToFit`, `clearSelection`.
- What to try:
1. Use each button and compare motion behavior.
2. Confirm clear-selection resets visual focus.

4. `js_04_layers_colorby`
- Purpose: separate "active layers" from "color source".
- Target APIs: `setLayers`, `setColorBy`.
- What to try:
1. Disable one layer and keep colorBy on the other.
2. Set colorBy to `base` and compare legend.

5. `js_05_runtime_mutation`
- Purpose: mutate overlays at runtime.
- Target APIs: `upsertLayer`, `patchLayerNodes`, `setLayerLabel`, `setColorRule`, `removeLayer`.
- What to try:
1. Move threshold slider.
2. Rename layer.
3. Apply color rule function.
4. Remove layer and revert to base.

6. `js_06_layout_slots`
- Purpose: embed UI components in external host divs.
- Target APIs: `mount.slots`, `setLayout`, `setUIVisibility`.
- What to try:
1. Toggle info/minimap visibility.
2. Hide/show toolbar chrome.
3. Inspect slot ownership behavior.

7. `js_07_events`
- Purpose: subscribe to viewer events from host code.
- Target APIs: `on`, `off` with `statechange`, `selectionchange`, `themechange`, `layoutchange`.
- What to try:
1. Trigger events with buttons.
2. Click `Unsubscribe Events`.
3. Confirm log no longer updates.

8. `js_08_compare_basics`
- Purpose: first compare orchestration flow.
- Target APIs: `FXGraphCompare.create`, `setSync`, `setCompact`, `setColumns`.
- What to try:
1. Toggle selection/theme sync.
2. Change compact mode.
3. Change compare columns.

### Level 2: Interesting Combinations

9. `adv_01_accuracy_dynamic`
- Purpose: real per-layer accuracy workflow with host controls.
- Target APIs: `setTheme`, `patchLayerNodes`, `setColorBy`, `selectNode`.
- What to try:
1. Adjust severity percentile.
2. Focus highest-severity node.

10. `adv_02_headless_slots_slider`
- Purpose: host-owned layout + slot embedding + dynamic recolor.
- Target APIs: `mount.slots`, `patchLayerNodes`, `setColorBy`.
- What to try:
1. Drag threshold slider.
2. Check info/minimap/legend in external panes.

11. `adv_03_fullscreen_toolbar`
- Purpose: fullscreen via both toolbar and direct API.
- Target APIs: `layout.fullscreen.button`, `enterFullscreen`, `exitFullscreen`.
- What to try:
1. Enter/exit fullscreen from side buttons.
2. Use taskbar fullscreen toggle too.

12. `adv_04_tiled_compare`
- Purpose: two-graph tiled compare with shared taskbar, layer sync, and merged info panel.
- Target APIs: `FXGraphCompare.create`, `layout.tiled`, `sharedTaskbar`, `sync.mode`, `setTiled`, `setSync`.
- What to try:
1. Click a node in either graph — observe sync and merged info panel.
2. Use shared taskbar to switch theme across both viewers.
3. Use sync mode dropdown to switch between "Sync by ID" and "Don't sync".
4. Toggle tiled layout on/off via the Tiled button.

### Level 3: Current Mixed Demo

13. `js_99_combo_mixed`
- Purpose: demonstrate a realistic mixed usage pattern.
- Target APIs: compare sync, runtime mutation, event subscriptions, theme control, camera APIs.
- What to try:
1. Toggle compare sync flags.
2. Move threshold slider.
3. Focus worst node.
4. Run scripted sequence.

### Qualcomm-only

13. `qualcomm_metadata`
- Purpose: inspect Qualcomm PTQ metadata beside rendered graph.
- Target APIs: `create` plus host metadata composition.

## How to Learn Effectively

1. Run one testcase at a time.
2. Edit JS pane in small changes and rerun.
3. Keep the "Target APIs" list in view while editing.
4. Move to the next level only after you can explain current behavior.

## Common Mistakes

1. Using `setColorBy(layer)` when that layer is not active.
2. Forgetting to call `init()` after `create()`.
3. Patching a layer that has not been added via `upsertLayer`.
4. Assuming compare sync covers all dimensions when only selection is enabled.
