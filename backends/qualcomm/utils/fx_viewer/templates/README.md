# PyTorch FX Graph Viewer JS (Split Modules)

## Description
This folder contains the split JavaScript runtime for the FX viewer used by `fx_viewer/exporter.py`.

Current JS implementation provides a **standalone, embeddable, and highly interactive HTML5 Canvas viewer** for visualizing large-scale **PyTorch FX computational graphs**.  
It is designed with a **modular architecture** to handle **thousands of nodes and edges** smoothly, without external dependencies.

The viewer renders a graph payload with:
- main canvas graph renderer
- minimap
- search
- info panel
- extension layer toggles/color-by

## User Interactions & UX Features

- **Interactive Canvas**  
  Users can drag to pan and use the mouse wheel to zoom.

- **Smart Minimap**  
  A collapsible right-sidebar minimap shows a bird’s-eye view of the entire graph.  
  Users can drag a viewport box within the minimap to pan.

- **Selection Mode**  
  Clicking a node or edge isolates its execution flow.  
  Immediate inputs/outputs are highlighted while unrelated branches are dimmed.

- **Info Panel**  
  Selecting or hovering over an element reveals PyTorch metadata  
  (tensor shape, dtype, target) in a scrollable panel with clickable links to jump to connected nodes.

- **Fuzzy Search**  
  A robust search bar allows querying nodes by ID, op type, or meta attributes,  
  featuring keyboard navigation and instant camera teleportation.

- **Theme Engine**  
  Seamless toggling between optimized **Light** and **Dark** mode palettes.

- **Extensibility**  
  Supports custom overlays via the Python `GraphExtension` API.  
  Users can toggle data layers on/off and switch coloring modes via the Layers Menu.


## Files and Responsibilities
- `themes.js`: shared theme tokens (`THEMES`).
- `graph_data_store.js`: payload normalization, adjacency, active virtual-node composition.
- `search_engine.js`: fuzzy search over active nodes.
- `view_controller.js`: state machine and interaction orchestration.
- `canvas_renderer.js`: primary graph rendering + mouse interactions.
- `minimap_renderer.js`: minimap rendering + navigation.
- `ui_manager.js`: taskbar/search/layers/info panel/legend DOM.
- `fx_graph_viewer.js`: top-level facade (`FXGraphViewer`) and layout shell.

## Script Load Order
Load in dependency order:

1. `themes.js`
2. `graph_data_store.js`
3. `search_engine.js`
4. `view_controller.js`
5. `canvas_renderer.js`
6. `minimap_renderer.js`
7. `ui_manager.js`
8. `fx_graph_viewer.js`

## Payload Contract
Expected input to `new FXGraphViewer(containerId, payload)`:

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

## Public JS API
Primary API is on `FXGraphViewer`:

- `new FXGraphViewer(containerId, payload)`: construct viewer in target container.
- `viewer.init()`: initialize thumbnail + first fit/position.
- `viewer.renderAll()`: force redraw canvas and minimap.
- `viewer.selectNode(nodeId)`: select + center on node.
- `viewer.search(query)`: run search and populate search UI.

## Embedding Example (Split JS)
```html
<div id="graph-viewer-container" style="width:100vw;height:100vh;"></div>
<script>const graphPayload = /* injected JSON */;</script>
<script src="./themes.js"></script>
<script src="./graph_data_store.js"></script>
<script src="./search_engine.js"></script>
<script src="./view_controller.js"></script>
<script src="./canvas_renderer.js"></script>
<script src="./minimap_renderer.js"></script>
<script src="./ui_manager.js"></script>
<script src="./fx_graph_viewer.js"></script>
<script>
  const viewer = new FXGraphViewer("graph-viewer-container", graphPayload);
  viewer.init();
  window.fxViewer = viewer;
</script>
```

## Maintenance Notes
- Keep module boundaries strict: avoid cross-calling renderers from each other; route through `ViewerController`/`FXGraphViewer`.
- Prefer payload compatibility over UI-only changes; Python exporter and JS contract must stay aligned.
- If adding a new feature, document:
  - state shape changes (`ViewerController.state`)
  - payload schema changes (`GraphDataStore`)
  - UX wiring (`UIManager` events)
