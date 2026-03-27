# fx_viewer JS Runtime

This folder contains the browser runtime used by `FXGraphExporter`.

## Runtime Model

1. `FXGraphViewer` is the facade and public API.
2. `ViewerController` is the interaction/state controller.
3. `GraphDataStore` owns base + extension data and composes active virtual nodes.
4. Renderers (`CanvasRenderer`, `MinimapRenderer`) paint from controller/store state.
5. `UIManager` is a state adapter for taskbar/search/layers/info/legend controls.
6. `FXGraphCompare` orchestrates multi-view compare: builds compare DOM, wires sync, owns lifecycle.
7. `FXCompareTaskbar` renders the optional shared taskbar above the compare grid.

## Files and Responsibilities

1. `themes.js`: shared theme tokens (`THEMES`).
2. `graph_data_store.js`: payload normalization, topology cache, virtual-node composition.
3. `search_engine.js`: fuzzy search over active nodes.
4. `view_controller.js`: state machine and interaction orchestration.
5. `canvas_renderer.js`: primary graph rendering + canvas interactions.
6. `minimap_renderer.js`: minimap rendering + minimap navigation.
7. `ui_manager.js`: taskbar/search/layers/info panel/legend DOM.
8. `fx_graph_viewer.js`: `FXGraphViewer` facade, `FXGraphCompare` orchestrator, `FXCompareTaskbar` shared taskbar.

## Public API

The canonical API reference is maintained in:
1. `backends/qualcomm/utils/fx_viewer/README.md` (`JS API (Runtime)` section)

This file focuses on runtime internals, file responsibilities, and script load order.

## Config Precedence

1. `mount.slots.*` (placement) has highest precedence.
2. Explicit `layout.*` overrides preset defaults.
3. `layout.preset` fills missing values.
4. Built-in defaults are last fallback.

## Compare View DOM and Ownership

`FXGraphCompare` builds its own DOM shell inside `layout.container` and moves viewer sub-elements into it. The viewer's own wrapper is hidden but not destroyed.

### DOM tree

```
layout.container
  .fx-compare-root          flex column; created by FXGraphCompare
    .fx-compare-taskbar     optional; created by FXCompareTaskbar when sharedTaskbar.enabled
    .fx-compare-grid        CSS grid, repeat(N, 1fr) columns
      .fx-compare-col       one per viewer; flex column
        .fx-compare-col-header
        .fx-compare-minimap-row   fixed height; viewer.minimapRenderer.container moved here
        .fx-compare-canvas-row    flex:1; viewer.mainArea moved here
    .fx-compare-info-bar    single shared merged info panel; full width
```

### Ownership rules

- `FXGraphCompare` creates `.fx-compare-root`, `.fx-compare-grid`, `.fx-compare-col`, `.fx-compare-info-bar`.
- `FXCompareTaskbar` creates `.fx-compare-taskbar` and prepends it to `.fx-compare-root`.
- Each viewer's `.fx-viewer-wrapper` is hidden (`display:none`) while compare is active.
- `viewer.mainArea` and `viewer.minimapRenderer.container` are moved into compare columns; all other viewer internals stay in the hidden wrapper.
- DOM snapshots (parent + nextSibling) are recorded before any move. `_teardownCompareDOM()` restores them on `destroy()`.

### Resize handling

- A `ResizeObserver` on each `.fx-compare-canvas-row` calls `viewer.canvasRenderer.resize()` + `renderAll()`.
- An initial `requestAnimationFrame` fires after `_buildCompareDOM()` for the first layout pass.
- `MinimapRenderer` has its own `ResizeObserver` on its container for deferred-visibility cases (observatory, collapsed sections).

### Interaction ownership

| Concern | Owner |
|---------|-------|
| Selection sync | `FXGraphCompare._wireSelectionSync()` — source-guarded, propagates via `viewer.selectNode()` |
| Theme sync | `FXCompareTaskbar` (shared taskbar) + `FXGraphCompare._wireStateSync()` (state events) |
| Layers / ColorBy | `FXCompareTaskbar` — union of all extension ids, per-viewer `setLayers()` |
| Zoom / Fullscreen | `FXCompareTaskbar` — calls viewer public API or `requestFullscreen()` on root |
| Column count | `FXGraphCompare.setColumns()` — updates grid CSS, triggers resize RAF |
| Merged info panel | `FXGraphCompare._updateMergedInfo()` — called after selection sync, renders diff table |

## Payload Contract

Runtime input payload:

```js
{
  base: {
    legend: [{ label, color }],
    nodes: [{ id, label, x, y, width, height, info, tooltip, fill_color? }],
    edges: [{ v, w, points? }]
  },
  extensions: {
    [extId]: {
      name: string,
      legend: [{ label, color }],
      nodes: {
        [nodeId]: {
          info?: object,
          tooltip?: string[],
          label_append?: string[],
          fill_color?: string
        }
      }
    }
  }
}
```

## Script Load Order

1. `themes.js`
2. `graph_data_store.js`
3. `search_engine.js`
4. `view_controller.js`
5. `canvas_renderer.js`
6. `minimap_renderer.js`
7. `ui_manager.js`
8. `fx_graph_viewer.js`

## Maintenance Notes

1. Keep module boundaries strict; route orchestration through controller/facade.
2. Preserve payload compatibility when adding UI/runtime features.
3. If state shape changes, update docs and relevant contracts in this folder.
4. `FXGraphCompare` must always restore viewer DOM on `destroy()` — never leave viewer.mainArea detached.
