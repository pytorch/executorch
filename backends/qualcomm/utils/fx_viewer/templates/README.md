# fx_viewer JS Runtime

This folder contains the browser runtime used by `FXGraphExporter`.

## Runtime Model

1. `FXGraphViewer` is the facade and public API.
2. `ViewerController` is the interaction/state controller.
3. `GraphDataStore` owns base + extension data and composes active virtual nodes.
4. Renderers (`CanvasRenderer`, `MinimapRenderer`) paint from controller/store state.
5. `UIManager` is a state adapter for taskbar/search/layers/info/legend controls.
6. `FXGraphCompare` orchestrates multi-view compare and synchronization.

## Files and Responsibilities

1. `themes.js`: shared theme tokens (`THEMES`).
2. `graph_data_store.js`: payload normalization, topology cache, virtual-node composition.
3. `search_engine.js`: fuzzy search over active nodes.
4. `view_controller.js`: state machine and interaction orchestration.
5. `canvas_renderer.js`: primary graph rendering + canvas interactions.
6. `minimap_renderer.js`: minimap rendering + minimap navigation.
7. `ui_manager.js`: taskbar/search/layers/info panel/legend DOM.
8. `fx_graph_viewer.js`: top-level facade (`FXGraphViewer`) and compare orchestration (`FXGraphCompare`).

## Public API

The canonical API reference is maintained in:
1. `backends/qualcomm/utils/fx_viewer/README.md` (`JS API (Runtime)` section)

This file focuses on runtime internals, file responsibilities, and script load order.

## Config Precedence

1. `mount.slots.*` (placement) has highest precedence.
2. Explicit `layout.*` overrides preset defaults.
3. `layout.preset` fills missing values.
4. Built-in defaults are last fallback.

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
