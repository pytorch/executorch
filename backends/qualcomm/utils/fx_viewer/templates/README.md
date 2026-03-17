# fx_viewer JS Runtime (RFC v1)

This folder contains the browser runtime used by `FXGraphExporter`.

## Runtime Model

1. `FXGraphViewer` is the facade and public API.
2. `ViewerController` is the interaction/state controller.
3. `GraphDataStore` owns base + extension data and composes active virtual nodes.
4. Renderers (`CanvasRenderer`, `MinimapRenderer`) paint from controller/store state.
5. `UIManager` is a state adapter for taskbar/search/layers/info/legend controls.
6. `FXGraphCompare` orchestrates multi-view compare and synchronization.

## Public API (Implemented)

Construction:
1. `FXGraphViewer.create(config)`

State/events:
1. `getState`, `setState`, `replaceState`, `batch`
2. `on`, `off`

Viewer actions:
1. `setTheme`, `setLayers`, `setColorBy`
2. `selectNode`, `clearSelection`, `search`
3. `zoomToFit`, `panToNode`, `animateToNode`
4. `setUIVisibility`, `setLayout`
5. `enterFullscreen`, `exitFullscreen`, `destroy`

Layer mutation:
1. `upsertLayer`, `removeLayer`, `patchLayerNodes`, `setLayerLabel`, `setColorRule`

Compare:
1. `FXGraphCompare.create({ viewers, layout, sync })`
2. `setColumns`, `setCompact`, `setSync`, `destroy`

## Config Precedence

1. `mount.slots.*` (placement) has highest precedence.
2. Explicit `layout.*` overrides preset defaults.
3. `layout.preset` fills missing values.
4. Built-in defaults are last fallback.

## Key UX Behaviors

1. API-driven changes reflect in UI controls (`theme/layers/colorBy` sync).
2. Host container resize triggers canvas resize (`ResizeObserver`).
3. Headless slots support custom HTML controls around GraphView.
4. Optional taskbar fullscreen button is enabled via `layout.fullscreen.button`.

## Script Load Order

1. `themes.js`
2. `graph_data_store.js`
3. `search_engine.js`
4. `view_controller.js`
5. `canvas_renderer.js`
6. `minimap_renderer.js`
7. `ui_manager.js`
8. `fx_graph_viewer.js`
