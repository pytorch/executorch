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

---

## FXGraphViewer API Reference

### `FXGraphViewer.create(config)`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `config.payload` | object | required | Graph payload (base + extensions) |
| `config.mount.root` | string\|HTMLElement | required | CSS selector or element for root container |
| `config.mount.slots.canvas` | string\|HTMLElement | — | External canvas mount |
| `config.mount.slots.toolbar` | string\|HTMLElement | — | External toolbar mount |
| `config.mount.slots.info` | string\|HTMLElement | — | External info panel mount |
| `config.mount.slots.minimap` | string\|HTMLElement | — | External minimap mount |
| `config.mount.slots.legend` | string\|HTMLElement | — | External legend mount |
| `config.layout.preset` | `'split'`\|`'compact'`\|`'headless'`\|`'custom'` | `'split'` | Layout preset |
| `config.layout.panels.sidebar.visible` | boolean | `true` | Show sidebar |
| `config.layout.panels.sidebar.width` | number | `500` | Sidebar width in px |
| `config.layout.panels.sidebar.resizable` | boolean | `true` | Allow sidebar resize |
| `config.layout.panels.sidebar.collapsible` | boolean | `true` | Double-click resizer to collapse |
| `config.layout.panels.info.visible` | boolean | `true` | Show info panel |
| `config.layout.panels.minimap.visible` | boolean | `true` | Show minimap |
| `config.layout.panels.minimap.height` | number | `240` | Minimap height in px |
| `config.layout.panels.minimap.resizable` | boolean | `true` | Allow minimap resize |
| `config.layout.panels.legend.visible` | boolean | `true` | Show legend overlay |
| `config.layout.fullscreen.enabled` | boolean | `true` | Allow fullscreen |
| `config.layout.fullscreen.button` | boolean | `false` | Show fullscreen button in taskbar |
| `config.ui.controls.toolbar` | boolean | `true` | Show/hide entire taskbar |
| `config.ui.controls.search` | boolean | `true` | Show/hide search input |
| `config.ui.controls.layers` | boolean | `true` | Show/hide layers button |
| `config.ui.controls.theme` | boolean | `true` | Show/hide theme selector |
| `config.ui.controls.legend` | boolean | `true` | Show/hide legend overlay |
| `config.ui.controls.zoomButtons` | boolean | `true` | Show/hide zoom-to-fit button |
| `config.ui.controls.fullscreenButton` | boolean | `false` | Show/hide fullscreen button |
| `config.ui.controls.highlightButton` | boolean | `true` | Show/hide ancestor/descendant highlight toggle |
| `config.state.theme` | string | `'light'` | Initial theme (`'light'`, `'dark'`, or custom) |
| `config.state.colorBy` | string | `'base'` | Initial color-by extension id or `'base'` |
| `config.state.activeExtensions` | string[] | `[]` | Initially active extension ids |
| `config.state.highlightAncestors` | boolean | `true` | Highlight ancestors/descendants on select |

### Preset defaults

| Preset | sidebar | minimap | info | toolbar | search | layers | theme |
|--------|---------|---------|------|---------|--------|--------|-------|
| `split` (default) | visible | visible | visible | on | on | on | on |
| `compact` | hidden | hidden | hidden | on | on | on | on |
| `headless` | hidden | hidden | hidden | off | off | off | off |
| `custom` | hidden | hidden | hidden | on | on | on | on |

### Public methods

```
viewer.init()                           Initial zoom-to-fit / position
viewer.renderAll()                      Re-render canvas + minimap
viewer.getState()                       Snapshot of current state
viewer.setState(patch)                  Update controller state
viewer.setTheme(name)                   Switch theme ('light'|'dark'|custom)
viewer.setLayers(layerIds[])            Set active extensions
viewer.setColorBy(layerId)              Set active color-by extension
viewer.selectNode(nodeId, opts?)        Select node; opts: { animate, center, k }
viewer.panToNode(nodeId)                Pan to node without selecting
viewer.animateToNode(nodeId, opts?)     Animated pan; opts: { k }
viewer.setUIVisibility(flags)           Show/hide individual controls at runtime
viewer.setLayout(layoutPatch)           Apply layout changes at runtime
viewer.upsertLayer(id, payload)         Add or update an extension layer
viewer.removeLayer(id)                  Remove an extension layer
viewer.patchLayerNodes(id, nodePatch)   Update node data in an extension
viewer.enterFullscreen()                Enter fullscreen (returns Promise)
viewer.exitFullscreen()                 Exit fullscreen (returns Promise)
viewer.destroy()                        Teardown all DOM, listeners, renderers
viewer.on(event, fn)                    Subscribe to event; returns unsubscribe fn
viewer.off(event, fn)                   Unsubscribe from event
FXGraphViewer.registerTheme(name, tokens)  Register a custom theme globally
```

### Events

```
viewer.on('selectionchange', (e) => { e.nodeId, e.prevNodeId, e.source })
viewer.on('statechange',     (e) => { e.prevState, e.nextState, e.source })
viewer.on('layoutchange',    (e) => { e.prevState, e.nextState, e.source })
viewer.on('hover',           (e) => { e.nodeId })
```

### Examples

**1. Minimal split viewer**
```js
const viewer = FXGraphViewer.create({
    payload: myPayload,
    mount: { root: '#my-container' },
});
viewer.init();
```

**2. Compact viewer** (no sidebar, no minimap)
```js
const viewer = FXGraphViewer.create({
    payload: myPayload,
    mount: { root: '#my-container' },
    layout: { preset: 'compact' },
});
viewer.init();
```

**3. Headless with external slots**
```js
const viewer = FXGraphViewer.create({
    payload: myPayload,
    mount: {
        root: '#root',
        slots: { canvas: '#canvas-div', minimap: '#minimap-div', info: '#info-div', legend: '#legend-div' },
    },
    layout: { preset: 'headless' },
});
viewer.init();
```

**4. Runtime layer mutation**
```js
viewer.upsertLayer('quant', quantPayload);
viewer.patchLayerNodes('quant', { node_0: { fill_color: '#f00' } });
viewer.setColorBy('quant');
```

---

## FXGraphCompare API Reference

### `FXGraphCompare.create(config)`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `config.viewers` | FXGraphViewer[] | required | Viewers to compare |
| `config.layout.container` | string\|HTMLElement | required | CSS selector or element; compare DOM built inside |
| `config.layout.columns` | number | `2` | Side-by-side column count |
| `config.layout.minimapHeight` | number | `180` | Minimap row height in px (same for all columns) |
| `config.layout.infoHeight` | number | `200` | Merged info bar max-height in px |
| `config.layout.canvasHeightRatio` | number | `0.7` | Reserved; canvas fills remaining space after minimap |
| `config.sharedTaskbar.enabled` | boolean | `false` | Opt-in shared taskbar above the grid |
| `config.sharedTaskbar.controls.theme` | boolean | `true` | Theme selector in shared taskbar |
| `config.sharedTaskbar.controls.layers` | boolean | `true` | Layers+colorBy dropdown in shared taskbar |
| `config.sharedTaskbar.controls.zoomFit` | boolean | `true` | Zoom-to-fit button in shared taskbar |
| `config.sharedTaskbar.controls.syncMode` | boolean | `true` | Sync mode selector in shared taskbar |
| `config.sharedTaskbar.controls.fullscreen` | boolean | `true` | Fullscreen button in shared taskbar |
| `config.sync.mode` | `'none'`\|`'id'`\|`'layer'` | `'id'` | Selection sync mode |
| `config.sync.layer` | string | `''` | Extension id (when `mode='layer'`) |
| `config.sync.field` | string | `''` | Info key to match on (when `mode='layer'`) |

### Container height requirement

The compare root uses `height: 100%` to fill its container. If the container has no explicit height (e.g. only `min-height`), the flex chain collapses. A `ResizeObserver` on the container automatically sets `height = 90vh` as a fallback when `offsetHeight < 100`, and switches back to `100%` if the container later gains an explicit height.

### Methods

```
compare.setColumns(n)       Update column count; triggers resize
compare.setSync(patch)      Update sync config; e.g. { mode: 'none' }
compare.destroy()           Restore all viewer DOM, disconnect observers, remove compare root
```

### Examples

**1. Minimal two-viewer compare** (no shared taskbar, sync by id)
```js
const compare = FXGraphCompare.create({
    viewers: [viewerA, viewerB],
    layout: { container: '#compare-host' },
    sync: { mode: 'id' },
});
```

**2. With shared taskbar**
```js
const compare = FXGraphCompare.create({
    viewers: [viewerA, viewerB],
    layout: { container: '#compare-host' },
    sharedTaskbar: { enabled: true },
});
```

**3. Layer-field sync**
```js
const compare = FXGraphCompare.create({
    viewers: [viewerA, viewerB],
    layout: { container: '#compare-host' },
    sync: { mode: 'layer', layer: 'quant', field: 'node_name' },
});
