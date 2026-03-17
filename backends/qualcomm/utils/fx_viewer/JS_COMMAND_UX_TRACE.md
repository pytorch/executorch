# JS Command -> UX Trace (Current Runtime)

This document records how each public command affects runtime behavior and user experience.

## FXGraphViewer Commands

1. `create(config)`
- Code: `backends/qualcomm/utils/fx_viewer/templates/fx_graph_viewer.js`
- Effect: resolves mounts, applies preset/layout precedence, initializes renderers/UI.
- UX: viewer appears with split/headless/custom shell and requested controls.

2. `getState()`
- Effect: returns snapshot (selection, camera, theme, layers, ui visibility, layout state).
- UX: host can inspect current viewer state and build synchronized custom controls.

3. `setState(patch)`
- Effect: applies patch to controller state; camera/search handled explicitly.
- UX: host can programmatically drive viewer interactions and visual state.

4. `replaceState(next)`
- Effect: resets canonical state fields and optional camera/search.
- UX: deterministic reset/restore experience for scripted debugging flows.

5. `setTheme(name)`
- Effect: updates `themeName`, re-themes DOM + canvas/minimap.
- UX: immediate light/dark (or registered custom) switch with consistent UI tokens.

6. `setLayers(layerIds)`
- Effect: changes active extensions and recomputes virtual graph.
- UX: overlay data appears/disappears without rebuilding page.

7. `setColorBy(layerId)`
- Effect: color source switches (`base` or extension).
- UX: node colors + legend update to chosen metric/category.

8. `selectNode(nodeId, opts)`
- Effect: selection state updates, info panel refreshes, optional animation/center.
- UX: focus jumps to selected node with contextual dependency highlighting.

9. `clearSelection()`
- Effect: clears selection/ancestor/descendant sets.
- UX: graph returns to neutral exploration mode.

10. `search(query)`
- Effect: updates search candidates and search UI.
- UX: fuzzy lookup and quick navigation through large graphs.

11. `zoomToFit()` / `panToNode()` / `animateToNode()`
- Effect: updates camera transform.
- UX: smooth camera navigation and context framing.

12. `setUIVisibility(flags)`
- Effect: hides/shows taskbar/search/layers/theme/legend/fullscreen control.
- UX: host can build focused views (for demos, report sections, compare mode).

13. `setLayout(layoutPatch)`
- Effect: mutates layout behavior (sidebar visibility/width, minimap visibility/height, etc).
- UX: dynamic transitions between split/compact presentation modes.

14. `upsertLayer/removeLayer/patchLayerNodes/setLayerLabel/setColorRule`
- Effect: mutates extension registry and refreshes active graph.
- UX: dynamic threshold sliders and runtime overlays work without export roundtrip.

15. `enterFullscreen/exitFullscreen`
- Effect: browser fullscreen API on root container.
- UX: one-click deep-focus graph inspection; also wired to optional taskbar button.

16. `on/off`
- Events: `statechange`, `selectionchange`, `themechange`, `layoutchange`, `error`.
- UX: host-side dashboards/custom widgets can stay synchronized.

## FXGraphCompare Commands

1. `FXGraphCompare.create({ viewers, layout, sync })`
- Effect: wires compare orchestration and optional container columns.
- UX: side-by-side analysis with centralized controls.

2. `setColumns(n)`
- Effect: updates compare container grid columns.
- UX: quick N-pane layout changes for different screen sizes.

3. `setCompact(bool)`
- Effect: toggles compact layout (sidebar/minimap/info visibility in viewers).
- UX: maximizes graph canvas area during compare.

4. `setSync({ selection, camera, theme, layers })`
- Effect: controls cross-view propagation dimensions.
- UX: can lock only needed dimensions (e.g., selection only).

## Key Internal Command Paths

1. UI commands -> controller
- Files: `ui_manager.js`, `view_controller.js`
- UX: taskbar controls always use same state pipeline as external JS.

2. Controller state -> store recompute
- File: `view_controller.js` -> `graph_data_store.js`
- UX: layer/color changes stay consistent in canvas, legend, minimap, info panel.

3. Container resize -> canvas resize
- File: `canvas_renderer.js`
- Mechanism: `ResizeObserver` + window resize.
- UX: resizable host panes keep rendering sharp and correctly scaled.
