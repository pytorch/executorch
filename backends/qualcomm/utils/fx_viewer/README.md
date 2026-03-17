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
1. `FXGraphCompare.create({ viewers, layout, sync })`
2. `setColumns`, `setCompact`, `setSync`, `destroy`

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
2. Advanced combos (`adv_01` ... `adv_03`).
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
