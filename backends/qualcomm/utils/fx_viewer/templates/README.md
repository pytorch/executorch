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

1. `runtime.js`: shared theme tokens (`THEMES`), event utilities (`fxOn`, `fxOffAll`, `fxEsc`).
2. `graph_data_store.js`: payload normalization, topology cache, virtual-node composition.
3. `search_engine.js`: fuzzy search over active nodes.
4. `view_controller.js`: state machine and interaction orchestration.
5. `canvas_renderer.js`: primary graph rendering + canvas interactions.
6. `minimap_renderer.js`: minimap rendering + minimap navigation.
7. `ui_manager.js`: taskbar/search/layers/info panel/legend DOM.
8. `fx_graph_viewer.js`: `FXGraphViewer` facade (CSS in `_injectStyles()`).
9. `compare.js`: `FXCompareTaskbar` + `FXGraphCompare` orchestrator.

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
| Viewer visibility | `FXGraphCompare.setViewerVisible(name, visible)` — show/hide graph columns |
| Follow selection | Sidebar toggle — controls auto-pan on selection change |

### Selection Sync Modes

`FXGraphCompare` propagates the primary selection from the source viewer to all other viewers using `_findSyncTarget`. The sidebar selector controls the active mode.

| Mode | Sidebar label | Behavior |
|------|--------------|----------|
| `'auto'` (default) | Auto (handle→id) | Tries `debug_handle` set-intersection first; falls back to node-ID match |
| `'id'` | ID only | Selects the node with the same id in each target viewer; no-op if absent |
| `'layer'` | Ext: \<extId\>.\<field\> | Matches by `extensions[layer].nodes[nodeId].info[field]` value equality; picks last in topo order on multiple matches |
| `'none'` | Don't sync | No cross-viewer selection propagation |

#### `debug_handle` normalization (`mode: 'auto'`)

`debug_handle` in `node.info` is `int` (scalar) or `int[]` (list, for fused nodes). The sync engine normalizes both to a `Set<int>` and uses **set intersection** to find matches:

- `int dh` → `{dh}` (if non-zero)
- `int[] dh` → `Set(dh.filter(x => x !== 0))`
- `null / 0 / []` → empty set (no match)

Two nodes match if their normalized sets have a non-empty intersection. When multiple target nodes match, the **last in topological order** is selected (highest `topo_index`).

This enables three mapping patterns:
- **1-to-1**: same handle on both sides → direct match.
- **1-to-many** (decomposed ops): one source node → multiple target nodes sharing the same handle (e.g. `linear` → `t + mm + add`). The last decomposed op is selected.
- **many-to-1** (fused ops): a fused node carries a union tuple handle `(h1, h2)`. Any source node whose handle intersects `{h1, h2}` will match the fused node.

#### Registering a sync key on an extension

To expose an extension field as an explicit sync option in the sidebar, call `set_sync_key` on the Python `GraphExtension`:

```python
ext = GraphExtension(id="my_ext", name="My Extension")
ext.add_node_data(node_id, {"debug_handle": 42, ...})
ext.set_sync_key("debug_handle")   # appears as "Ext: my_ext.debug_handle" in sidebar
```

The sidebar will show `Ext: my_ext.debug_handle` as a selectable option. Selecting it activates `mode: 'layer'` with `layer='my_ext'` and `field='debug_handle'`.

### `compare.setViewerVisible(name, visible)`

Show or hide a graph column by name. Equivalent to checking/unchecking in the Graphs menu.

- `name` {string} — viewer name as passed to `FXGraphCompare.create()`
- `visible` {boolean} — `true` to show, `false` to hide

When showing a viewer that has a corresponding node for the current selection in another viewer, the newly visible viewer will pan to that node automatically.

### Follow Selection toggle

The sidebar includes a "Follow" toggle button. When enabled (\u2299), every selection change auto-pans all viewers to the corresponding node. When disabled (\u25cb), selections still sync but viewers don't auto-pan (user can navigate freely). Newly re-enabled viewers always pan to the active selection regardless of this toggle.

## Payload Contract

Runtime input payload:

```js
{
  base: {
    legend: [{ label, color }],
    nodes: [{ id, label, x, y, width, height, info, tooltip, fill_color? }],
    //   info.debug_handle: int | int[]  — present when generate_missing_debug_handles() was called
    edges: [{ v, w, points? }]
  },
  extensions: {
    [extId]: {
      name: string,
      legend: [{ label, color }],
      sync_keys: string[],   // fields registered via ext.set_sync_key(); drives sidebar options
      nodes: {
        [nodeId]: {
          info?: object,       // arbitrary key/value; info.debug_handle used by 'auto' sync
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

1. `runtime.js`
2. `graph_data_store.js`
3. `search_engine.js`
4. `view_controller.js`
5. `canvas_renderer.js`
6. `minimap_renderer.js`
7. `ui_manager.js`
8. `fx_graph_viewer.js`
9. `compare.js`

## Maintenance Notes

1. Keep module boundaries strict; route orchestration through controller/facade.
2. Preserve payload compatibility when adding UI/runtime features.
3. If state shape changes, update docs and relevant contracts in this folder.
4. `FXGraphCompare` must always restore viewer DOM on `destroy()` — never leave viewer.mainArea detached.

---

## Runtime Internals

### GraphDataStore

Manages the raw JSON graph payload and constructs the "Virtual Node" topology.

**State:** `baseData` (structural nodes/edges), `extensions` (annotation overlays), `activeNodes` (pre-computed flat array), `activeNodeMap` (O(1) id→node), `adjList`/`revAdjList` (edge traversal), `graphBounds` (camera zoom target).

**Topology Init (`_initTopology`):** Loops over `baseData.nodes` once to calculate global bounds. Normalizes coordinates so the top-left node starts at (50, 50). Builds adjacency lists.

**Virtual Node Composition (`computeActiveGraph`):** For each base node, creates a flat `info` dict. Iterates active extension ids; if an extension has data for the node, prefixes keys (e.g. `Profiler.latency: 15`) and merges them. Concatenates `label_append` and `tooltip` arrays. Resolves `fill_color` from `colorById`. Pre-computing `activeNodes` on checkbox toggle avoids GC pressure during the 60FPS render loop.

**Traversal:** `getAncestors(id)` / `getDescendants(id)` — BFS over `revAdjList`/`adjList` for canvas selection highlighting.

---

### SearchEngine

Fuzzy search over active graph nodes with token scoring and context highlighting.

**Algorithm:**
1. Tokenize query by spaces (e.g. `"conv 15ms"` → `["conv", "15ms"]`).
2. Iterate `activeNodes`. Because `node.info` is a flattened dict with extension prefixes, the engine searches extension data natively.
3. Scoring: `node.id` match = +10, `op` = +5, `target` = +3, other key/value = +1.
4. Context highlighting: wraps matched substring in `<span style="background: yellow">` so the dropdown shows exactly why a node matched.
5. Filter to nodes matching the maximum number of tokens (fuzzy AND).

---

### ViewerController

Centralized state machine managing interactions, camera transforms, selections, and extension visibility.

**State fields:** `hoveredNodeId`, `hoveredEdge`, `selectedNodeId`, `selectedEdge`, `previewNodeId`, `ancestors`/`descendants` (Sets), `searchCandidates`, `searchSelectedIndex`, `highlightAncestors`, `themeName`, `activeExtensions` (Set), `colorBy`, `highlightGroups` (Map<groupId, {nodeIds: Set<string>, color: string}>).

**`setState(patch)`:** Merges patch. If `activeExtensions` or `colorBy` changed, calls `store.computeActiveGraph()`, regenerates minimap thumbnail, updates legend and info panel. If `themeName` changed, calls `ui.applyThemeToDOM()`. Always calls `viewer.renderAll()` and emits `statechange`.

**`animateToTransform(x, y, k)`:** Uses `requestAnimationFrame` with easeOutCubic over 300ms to interpolate camera position.

**`zoomToFit()`:** If a node is selected, collects its 2-hop neighborhood; if an edge is selected, collects 1-hop neighbors of both endpoints. Computes bounding box and animates camera to fit. Falls back to full graph bounds if nothing is selected.

**Selection:** `selectNode` / `selectEdge` run BFS ancestors/descendants via `store`, update state, and call `ui.updateInfoPanel` / `ui.updateEdgeInfoPanel`. `clearSelection` nullifies all selection state and hides the info panel.

**Search flow:** `handleSearch` → `SearchEngine.search` → `setState({searchCandidates})` → `ui.updateSearchResults`. Arrow keys call `handleSearchNavigate` (pan preview). Enter/click calls `handleSearchSelect` (full select + close menu).

---

### CanvasRenderer

High-performance 2D canvas rendering of the main graph with pan/zoom and hover interactions.

**Coordinate spaces:** DOM mouse events are in Screen Space. Nodes/edges live in Graph Space. Conversion: `graphX = (screenX - transform.x) / transform.k`. Device pixel ratio (`dpr`) is applied via `ctx.scale(dpr, dpr)` to prevent blurring on retina displays.

**Pan/zoom:** `mousedown`/`mousemove` delta → `transform.x/y`. `wheel` → exponential `zoomFactor`, pivot at cursor: `transform.x = mouseX - graphX * newK`.

**`render()` loop:**
1. Clear canvas, fill theme background.
2. Apply `ctx.scale(dpr, dpr)`, `ctx.translate(transform.x, y)`, `ctx.scale(k, k)`.
3. Compute `opacity = 0.15` for nodes/edges outside the active selection ancestry.
4. Draw edges (with midpoint tensor-shape labels), then node rectangles with multi-line labels.
5. **Highlight group overlay pass** (after node rendering): iterates `state.highlightGroups`; for each group draws a thick 6px solid border outside the node rect (offset by `borderWidth/2` so it does not clip node fill or text). Multiple groups coexist; last group in Map iteration order wins visually on overlapping nodes.

**`drawSmartTooltip()`:** Tests 4 candidate positions (up/down/left/right) against viewport bounds. Prefers "right" if it fits. Draws a dashed connector line scaled by `1/transform.k`.

**Dynamic color:** `shadeColor()` lightens/darkens custom extension fill colors for hover/selected/ancestor states, preserving analytical heatmap context.

---

### MinimapRenderer

Minimap overview rendering with viewport tracking and click/drag navigation.

**Coordinate transforms:**
- `minimapScale = min(mw/graphW, mh/graphH) * 0.9` — shrinks Graph Space to Minimap Space.
- `thumbnailOffset` — centers the scaled graph in the minimap container.
- Click/drag: `graphX = (screenX - thumbnailOffset.x) / minimapScale` → update `transform.x/y` to center main canvas on that point.
- Viewport rect: `viewX = -transform.x / transform.k`, projected to minimap: `mx = viewX * minimapScale + thumbnailOffset.x`.

**`generateThumbnail()`:** Draws the full graph to an off-screen canvas buffer. Called only when graph data or theme changes.

**`render()`:** Blits thumbnail (O(1)), then overlays search candidate dots, ancestor/descendant highlights, and the red viewport rectangle.

---

### UIManager

Manages all non-canvas DOM elements: taskbar, search, layers dropdown, legend overlay, and info panel.

**`buildUI()`:** Constructs HTML overlay components programmatically. Attaches `input`/`keydown` listeners on search input, relaying to `ViewerController`.

**Search rendering:** `updateSearchResults` renders 50 items initially. `onscroll` listener appends 20 more when near the bottom (chunked rendering to prevent DOM freeze).

**Layers menu:** Reads `viewer.store.extensions` to build checkboxes (`activeExtensions`) and radio buttons (`colorBy`). `onchange` calls `controller.setState`.

**Info panel (`updateInfoPanel`):**
1. Renders core PyTorch keys (`op`, `name`, `target`, `args`, `kwargs`, `shape`, `dtype`) at top.
2. Renders Inputs/Outputs as clickable `fx-link` elements that animate camera to related nodes.
3. Groups remaining prefixed keys (e.g. `Profiler.latency`) by prefix, rendering section headers.

**Legend (`renderLegend`):** Reads `colorBy` state, fetches legend array from `store`, renders color swatches with `shadeColor` adjustment for dark mode.

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
viewer.addHighlightGroup(groupId, nodeIds, color)  Add/replace a named highlight group overlay
viewer.removeHighlightGroup(groupId)               Remove a named highlight group
viewer.clearAllHighlightGroups()                   Remove all highlight groups
viewer.getHighlightGroups()                        Returns Map<groupId, {nodeIds, color}>
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

## Highlight Groups

Highlight groups are a **read-only programmatic overlay** separate from the single-node selection system. They are set via the JS API and render as thick colored borders around specified nodes.

### Key properties

- Multiple named groups coexist simultaneously on the same viewer.
- Each group has a `groupId` (string), a set of `nodeIds`, and a CSS `color`.
- Rendering: 6px solid border drawn **outside** the node rect (offset by `borderWidth/2`) so it does not clip node fill or text.
- Groups survive `clearSelection()` — they are independent of selection state.
- Groups are drawn as a separate pass **after** all node rendering, so they always appear on top.

### API

```js
// Create or replace a named group
viewer.addHighlightGroup(groupId, nodeIds, color)
// groupId: string — unique name
// nodeIds: string[] — node IDs to highlight
// color: string — CSS color, e.g. '#ff6600' or 'rgba(255,100,0,0.8)'

// Remove one group
viewer.removeHighlightGroup(groupId)

// Remove all groups
viewer.clearAllHighlightGroups()

// Read current groups (returns a shallow copy)
viewer.getHighlightGroups()  // → Map<string, {nodeIds: Set<string>, color: string}>
```

### Example

```js
// Highlight two groups with different colors
viewer.addHighlightGroup('critical', ['conv_0', 'conv_1'], '#ff0000');
viewer.addHighlightGroup('attention', ['attn_q', 'attn_k', 'attn_v'], '#0066ff');

// Remove one group
viewer.removeHighlightGroup('critical');

// Clear all
viewer.clearAllHighlightGroups();
```

### Compare mode

`FXGraphCompare` does **not** automatically propagate highlight groups across viewers. Groups are set explicitly per-viewer via the JS API. The compare sync only propagates the primary selection (unchanged).

### State storage

`highlightGroups` is stored on `ViewerController.state.highlightGroups` as a `Map<string, {nodeIds: Set<string>, color: string}>`. It is included in `snapshotState()` (shallow copy). `setState({ highlightGroups: newMap })` replaces the reference atomically and triggers a re-render.

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
| `config.sync.mode` | `'none'`\|`'id'`\|`'auto'`\|`'layer'` | `'auto'` | Selection sync mode |
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
