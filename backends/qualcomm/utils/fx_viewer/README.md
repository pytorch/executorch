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
1. `FXGraphCompare.create({ viewers, layout, sharedTaskbar, sync })`
2. `setColumns(n)`, `setSync(patch)`, `destroy()`
3. `setTiled()` / `setCompact()` — no-ops (tiled layout is always used in compare mode)
4. `sharedTaskbar.enabled` — opt-in (`false` by default); when `true`, injects a shared taskbar above the grid
5. `sharedTaskbar.controls`: `{ theme, layers, zoomFit, fullscreen, syncMode }` — all default `true` when taskbar is enabled
6. `sync.mode`: `'none' | 'id' | 'layer'`; when `'layer'`: also set `sync.layer` (extension id) and `sync.field` (info key)
7. `layout.container`: CSS selector or `HTMLElement` — required; `FXGraphCompare` builds its DOM inside this element
8. `layout.columns`: number of side-by-side viewer columns (default `2`)
9. `layout.minimapHeight`: minimap row height in px (default `180`)
10. `layout.infoHeight`: merged info bar max-height in px (default `200`)

## Compare View Architecture

`FXGraphCompare` owns the compare layout DOM entirely. It builds a structured shell inside `layout.container` and moves canvas/minimap elements out of each viewer's own wrapper into that shell.

### DOM Structure

```
layout.container  (user-supplied div)
  .fx-compare-root  (flex column, fills container — created by FXGraphCompare)
    .fx-compare-taskbar  (flex row — only present when sharedTaskbar.enabled)
    .fx-compare-grid  (CSS grid, repeat(N, 1fr) columns)
      .fx-compare-col  (one per viewer, flex column)
        .fx-compare-col-header  (graph title label)
        .fx-compare-minimap-row  (fixed height — same across all columns)
          viewer.minimapRenderer.container  (moved here from viewer.sidebar)
        .fx-compare-canvas-row  (flex: 1 — same height across all columns)
          viewer.mainArea  (moved here from viewer.wrapper)
    .fx-compare-info-bar  (single shared merged info panel, spans full width)
```

Each viewer's own `.fx-viewer-wrapper` (sidebar, resizer, etc.) is hidden (`display: none`) while compare is active. The viewer's public API (`setTheme`, `selectNode`, `renderAll`, etc.) continues to work normally because it operates on `mainArea` and `minimapRenderer.container` regardless of where they are in the DOM.

### Uniform Row Heights

All minimap rows are the same fixed height (`layout.minimapHeight`). All canvas rows share `flex: 1` inside the same flex column, so they expand to fill identical remaining space. Vertical boundaries are aligned across graphs because the columns are siblings in the same CSS grid row — no per-column height negotiation needed.

### Ownership and Lifecycle

1. **`FXGraphCompare` owns the compare DOM.** It creates `.fx-compare-root`, `.fx-compare-grid`, `.fx-compare-col` elements and appends them to `layout.container`.
2. **Viewers own their renderers.** `FXGraphCompare` only moves `viewer.mainArea` and `viewer.minimapRenderer.container` — it does not touch canvas contexts, event listeners, or state machines.
3. **DOM snapshots for teardown.** Before moving any element, `FXGraphCompare` records its original parent and next sibling in a `WeakMap`. `destroy()` calls `_teardownCompareDOM()` which restores every element to its original position and un-hides each viewer wrapper.
4. **`FXCompareTaskbar` is instantiated by `FXGraphCompare`** when `sharedTaskbar.enabled` is true. It prepends its element to `.fx-compare-root` (not to `layout.container`), so it sits above the grid inside the flex column. `FXCompareTaskbar.destroy()` is called by `FXGraphCompare.destroy()`.
5. **Canvas resize.** A `ResizeObserver` is attached to each `.fx-compare-canvas-row`. When the row resizes (window resize, column count change), it calls `viewer.canvasRenderer.resize()` + `viewer.renderAll()`. An initial `requestAnimationFrame` resize fires after `_buildCompareDOM()` to handle the first layout pass.

### Interaction Control

| Action | Owner | Mechanism |
|--------|-------|-----------|
| Node selection sync | `FXGraphCompare._wireSelectionSync()` | Listens to `viewer.on('selectionchange')`; propagates via `viewer.selectNode()` with source guard to prevent loops |
| Theme sync (shared taskbar) | `FXCompareTaskbar` | Calls `viewer.setTheme()` on all viewers directly |
| Theme sync (state change) | `FXGraphCompare._wireStateSync()` | Listens to `viewer.on('statechange')`; propagates theme changes to other viewers |
| Layers / ColorBy | `FXCompareTaskbar` | Builds union of all extension ids; calls `viewer.setLayers()` per viewer |
| Zoom to Fit | `FXCompareTaskbar` | Calls `viewer.controller.zoomToFit()` on all viewers |
| Fullscreen | `FXCompareTaskbar` | Calls `requestFullscreen()` on `.fx-compare-root` |
| Column count | `FXGraphCompare.setColumns()` | Updates `_grid.style.gridTemplateColumns`; triggers resize RAF |
| Sync mode | `FXCompareTaskbar` → `FXGraphCompare.setSync()` | Updates `this.sync`; next selection event uses new mode |
| Merged info panel | `FXGraphCompare._updateMergedInfo()` | Called after selection sync; renders a diff table into `.fx-compare-info-bar` |

### Selection Sync Modes

- `mode: 'none'` — no cross-viewer selection propagation.
- `mode: 'id'` (default) — selects the node with the same id in each other viewer; no-op if id is absent.
- `mode: 'layer'` — matches by `extensions[layer].nodes[nodeId].info[field]` value; on multiple matches, picks the node with the highest `topo_index` in that extension (or last in `activeNodes` order if no topo index).

### Merged Info Panel

When a node is selected (and sync propagates), `_updateMergedInfo(nodeIdMap)` renders a comparison table into `.fx-compare-info-bar`:
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

## Color Rules

Available rules:
1. `CategoricalColorRule(attribute, color_map=None)`
2. `NumericColorRule(attribute, cmap="viridis", handle_outliers=True)`

Rule selection:
1. Use categorical for discrete semantic labels.
2. Use numeric for continuous measured metrics.
3. Keep `handle_outliers=True` for noisy distributions.
4. For rank/index-like metrics, set `handle_outliers=False`.

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
