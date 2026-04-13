# fx_viewer

`fx_viewer` exports FX graphs to interactive HTML and provides an embeddable JavaScript runtime.

## What It Provides

Python side:
1. Extract FX graph (`torch.export` / `torch.fx`).
2. Compute layout (Grandalf/Sugiyama).
3. Build payload (`base` + `extensions`).
4. Export JSON / JS snippet / standalone HTML.

JS side:
1. Canvas graph + minimap + info panel + search.
2. Layer toggles and color-by controls.
3. State-driven API for embedding, compare mode, fullscreen, and runtime layer mutation.

## Quick Start

From repo root:

```bash
source .venv/bin/activate
python backends/qualcomm/utils/fx_viewer/examples/demo_fx_viewer_extensions.py --model both
```

Outputs:
1. `swin_graph_v3_extensions.html`
2. `llama_graph_v3_extensions.html`

## Python API

```python
from executorch.backends.qualcomm.utils.fx_viewer import (
    FXGraphExporter,
    GraphExtension,
    CategoricalColorRule,
)

exporter = FXGraphExporter(graph_module)

ext = GraphExtension(id="backend", name="Backend Assignment")
ext.add_node_data("node_0", {"backend": "cpu"})
ext.set_color_rule(CategoricalColorRule(attribute="backend"))

exporter.add_extension(ext)
exporter.export_html("graph.html")
```

Main exporter methods:
1. `generate_json_payload()`
2. `export_json(path)`
3. `export_js(container_id)`
4. `export_html(path)`

Python tutorial:
1. `backends/qualcomm/utils/fx_viewer/examples/PYTHON_API_TUTORIAL.md`

## JS API (Runtime)

Construction:
1. `FXGraphViewer.create(config)`
2. Compatibility constructor: `new FXGraphViewer(containerId, payload)`

State/events:
1. `getState`, `setState`, `replaceState`, `batch`
2. `on`, `off`

Viewer actions:
1. `setTheme`, `setLayers`, `setColorBy`
2. `selectNode`, `clearSelection`, `search`
3. `zoomToFit`, `panToNode`, `animateToNode`
4. `setUIVisibility`, `setLayout`
5. `enterFullscreen`, `exitFullscreen`, `destroy`

Runtime layer mutation:
1. `upsertLayer`, `removeLayer`, `patchLayerNodes`, `setLayerLabel`, `setColorRule`

Compare:
1. `FXGraphCompare.create({ viewers, layout, sync })` — `viewers` accepts `FXGraphViewer[]` or `Map<name, FXGraphViewer>`
2. `setSync(patch)`, `destroy()`
3. `setTiled()` / `setCompact()` — no-ops (tiled layout is always used in compare mode)
4. `sync.mode`: `'auto'` (default) | `'id'` | `'layer'` | `'none'`
   - `'auto'`: tries `debug_handle` set-intersection first, falls back to node-ID match
   - `'id'`: matches by node id only
   - `'layer'`: matches by `extensions[layer].nodes[nodeId].info[field]`; set `sync.layer` and `sync.field`
   - `'none'`: no sync
5. `layout.container`: CSS selector or `HTMLElement` — required

Highlight groups (programmatic overlay, independent of selection):
1. `viewer.addHighlightGroup(groupId, nodeIds, color)` — add/replace a named group
2. `viewer.removeHighlightGroup(groupId)` — remove one group
3. `viewer.clearAllHighlightGroups()` — remove all groups
4. `viewer.getHighlightGroups()` — returns `Map<groupId, {nodeIds, color}>`

## Compare View Architecture

`FXGraphCompare` owns the compare layout DOM entirely. It builds a structured shell inside `layout.container` and moves canvas/minimap elements out of each viewer's own wrapper into that shell.

### DOM Structure

```
layout.container  (user-supplied div)
  .fx-compare-root  (flex column, fills container — created by FXGraphCompare)
    .fx-compare-grid  (CSS grid: 160px sidebar + N×1fr graph columns, 3 rows)
      .fx-compare-sidebar-cell  (col 1, rows 1-2 — shared controls)
      .fx-compare-minimap-cell  (col i+2, row 1 — one per viewer)
        viewer.minimapRenderer.container  (moved here from viewer.sidebar)
        .fx-compare-graph-name  (graph title label, absolute overlay)
      .fx-compare-canvas-cell  (col i+2, row 2 — one per viewer)
        viewer.mainArea  (moved here from viewer.wrapper)
      .fx-compare-info-row  (col 1..-1, row 3 — CSS subgrid, merged info panel)
        .fx-compare-sidebar-info-cell  (col 1 — empty spacer)
        .fx-compare-info-hdr  (col i+2 — graph name header, one per visible viewer)
        .fx-compare-info-prop  (col 1 — property name, sticky left)
        .fx-compare-info-val  (col i+2 — property value, one per visible viewer)
```

Each viewer's own `.fx-viewer-wrapper` (sidebar, resizer, etc.) is hidden (`display: none`) while compare is active. The viewer's public API (`setTheme`, `selectNode`, `renderAll`, etc.) continues to work normally because it operates on `mainArea` and `minimapRenderer.container` regardless of where they are in the DOM.

### Uniform Row Heights

All minimap cells are the same fixed height (CSS `minmax(100px, 200px)`). All canvas cells share `minmax(50vh, 100vh)` in the same grid row, so they expand to fill identical space. Vertical boundaries are aligned across graphs because the cells are siblings in the same CSS grid — no per-column height negotiation needed.

### Ownership and Lifecycle

1. **`FXGraphCompare` owns the compare DOM.** It creates `.fx-compare-root`, `.fx-compare-grid`, `.fx-compare-sidebar-cell`, `.fx-compare-minimap-cell`, `.fx-compare-canvas-cell`, and `.fx-compare-info-row` elements and appends them to `layout.container`.
2. **Viewers own their renderers.** `FXGraphCompare` only moves `viewer.mainArea` and `viewer.minimapRenderer.container` — it does not touch canvas contexts, event listeners, or state machines.
3. **DOM snapshots for teardown.** Before moving any element, `FXGraphCompare` records its original parent and next sibling in a `WeakMap`. `destroy()` calls `_teardownCompareDOM()` which restores every element to its original position and un-hides each viewer wrapper.
4. **Canvas resize.** A `ResizeObserver` is attached to each `.fx-compare-canvas-cell`. When the cell resizes (window resize, column visibility change), it calls `viewer.canvasRenderer.resize()` + `viewer.renderAll()`. An initial `requestAnimationFrame` resize fires after `_buildCompareDOM()` to handle the first layout pass.

### Interaction Control

| Action | Owner | Mechanism |
|--------|-------|-----------|
| Node selection sync | `FXGraphCompare._wireSelectionSync()` | Listens to `viewer.on('selectionchange')`; propagates via `viewer.selectNode()` with source guard to prevent loops |
| Theme sync (state change) | `FXGraphCompare._wireStateSync()` | Listens to `viewer.on('statechange')`; propagates theme changes to other viewers; calls `_applyCompareTheme()` |
| Theme (compare shell) | `FXGraphCompare._applyCompareTheme()` | Sets CSS custom properties on `.fx-compare-root`; styles sidebar controls inline |
| Layers / ColorBy | Sidebar Layers button | Builds union of all extension ids; calls `viewer.setLayers()` / `viewer.setColorBy()` per viewer |
| Zoom to Fit | Sidebar Fit button | Calls `viewer.controller.zoomToFit()` on all viewers |
| Fullscreen | Sidebar Full button | Calls `requestFullscreen()` on `.fx-compare-root`; `fullscreenchange` listener updates button icon |
| Sync mode | Sidebar sync selector → `FXGraphCompare.setSync()` | Updates `this.sync`; next selection event uses new mode |
| Merged info panel | `FXGraphCompare._updateMergedInfo()` | Called after selection sync; renders a diff table into `.fx-compare-info-row` |

### Selection Sync Modes

| Mode | Sidebar label | Behavior |
|------|--------------|----------|
| `'auto'` (default) | Auto (handle→id) | `debug_handle` set-intersection first; falls back to node-ID match |
| `'id'` | ID only | Matches by node id; no-op if absent |
| `'layer'` | Ext: \<extId\>.\<field\> | Matches by extension field value; picks last in topo order on multiple matches |
| `'none'` | Don't sync | No propagation |

`debug_handle` normalization: `int` → `{int}`, `int[]` → `Set(int[])`, `null/0/[]` → empty set. Two nodes match if their sets have a non-empty intersection. The **last in topological order** is selected on multiple matches.

Three mapping patterns:
- **1-to-1**: same handle on both sides.
- **1-to-many** (decomposed ops): `linear` → `t + mm + add`, all share the same handle; last decomposed op is selected.
- **many-to-1** (fused ops): fused node carries union tuple handle `(h1, h2)`; any source node whose handle intersects `{h1, h2}` matches.

### Merged Info Panel

When a node is selected (and sync propagates), `_updateMergedInfo(nodeIdMap)` renders a comparison table into `.fx-compare-info-row`:
- Header row: "Property" | Graph 1 | Graph 2 | ...
- One row per property (union of all `node.info` keys across all selected nodes)
- Rows where values differ across graphs are highlighted amber (`.fx-diff`)
- Missing values shown as `—`

## Canonical Data Contract

Top-level payload:
1. `base`: `{ legend, nodes, edges }`
2. `extensions`: map keyed by extension id

`base.nodes[]` fields:
1. `id`, `label`, `x`, `y`, `width`, `height`
2. `info`: metadata used by search/info panel
3. `tooltip`: base tooltip lines
4. `fill_color` (optional)

`base.edges[]` fields:
1. `v`, `w`
2. `points` (optional routed polyline)

## Extension Authoring Guide

Key contract:
1. Add extension data explicitly with `add_node_data(node_id, data)`.
2. Formatter input is exactly that stored `data` dictionary.
3. Formatters must return `list[str]`.

What formatters do not receive implicitly:
1. Full FX node object.
2. Base graph `info` fields.
3. Global graph context.

If you need base attributes (for example `target`, `op`) in extension label/tooltip,
copy them into extension data before formatter use.

### Sync key registration

To expose an extension field as an explicit sync option in the compare sidebar:

```python
ext.set_sync_key("debug_handle")
```

This makes `Ext: <ext_id>.debug_handle` appear as a selectable option in the compare sidebar. Selecting it activates `mode: 'layer'` with that extension and field.

The `per_layer_accuracy` extension automatically registers `debug_handle` as a sync key when built via `_add_accuracy_extension`.

## Color Rules

Available rules:
1. `CategoricalColorRule(attribute, color_map=None)`
2. `NumericColorRule(attribute, cmap="viridis", handle_outliers=True)`

Rule selection:
1. Use categorical for discrete semantic labels.
2. Use numeric for continuous measured metrics.
3. Keep `handle_outliers=True` for noisy distributions.
4. For rank/index-like metrics, set `handle_outliers=False`.

## 3-Graph Compare Demo

Standalone demo showing all three `debug_handle` mapping patterns in one compare view:

```bash
python backends/qualcomm/utils/fx_viewer/examples/demo_3graph_compare.py
```

Output: `demo_3graph_compare.html`

Three graphs:
1. **Reference (float)**: unique int handle per node.
2. **Decomposed (1→many)**: each `linear` → `t + mm + add`, all three share the same handle.
3. **Fused (many→1)**: `relu` nodes that follow a `linear` carry a union tuple handle `(linear_h, relu_h)`.

Expected sync behavior (mode `auto`, set intersection):
- Click `linear` (handle `{6}`) in Graph 1 → Graph 2: `add_tensor` (last of `{t,mm,add}`). Graph 3: `relu` (handle `{6,7}`).
- Click `relu` (handle `{7}`) in Graph 1 → Graph 2: `relu_default`. Graph 3: `relu` (handle `{6,7}`).
- Click `relu` (handle `{6,7}`) in Graph 3 → Graph 1: `relu` (last among `{linear,relu}`). Graph 2: `relu_default`.

## Unified API Harness

Files:
1. Generator: `backends/qualcomm/utils/fx_viewer/examples/generate_api_test_harness.py`
2. Template: `backends/qualcomm/utils/fx_viewer/examples/harness_template.html`
3. Testcases: `backends/qualcomm/utils/fx_viewer/examples/harness_testcases.py`
4. Tutorial testcase guide: `backends/qualcomm/utils/fx_viewer/examples/FX_VIEWER_API_TESTCASES.md`

Generate harnesses:

```bash
source .venv/bin/activate
export PYTHONPATH=~/:$PYTHONPATH
python backends/qualcomm/utils/fx_viewer/examples/generate_api_test_harness.py
```

Generated outputs:
1. `fx_viewer_api_test_harness_portable.html`
2. `fx_viewer_api_test_harness_qualcomm.html`

Suggested learning order:
1. JS beginner ladder (`js_01` ... `js_08` in testcase guide).
2. Advanced combos (`adv_01` ... `adv_04`).
3. Final mixed demo (`js_99_combo_mixed`).

## Testing

Contract tests:
1. `tests/test_exporter_contract.py`

Run:

```bash
source .venv/bin/activate
pytest -q tests/test_exporter_contract.py
```

## References

1. API RFC: `backends/qualcomm/utils/fx_viewer/RFC_FX_VIEWER_API_INTERFACE.md`
2. Implementation status: `backends/qualcomm/utils/fx_viewer/RFC_API_IMPLEMENTATION_STATUS.md`
3. JS runtime internals: `backends/qualcomm/utils/fx_viewer/templates/README.md`
