# fx_viewer

`fx_viewer` exports FX graphs to interactive HTML and provides a state-driven JS runtime.

## What It Provides

Python side:
1. Extract FX graph (`torch.export` / `torch.fx`).
2. Compute layout (Grandalf/Sugiyama).
3. Build payload (`base` + `extensions`).
4. Export JSON / JS snippet / standalone HTML.

JS side:
1. Canvas graph + minimap + info panel + search.
2. Layer toggles and color-by.
3. State-driven API for embedding, compare, fullscreen, and runtime layer mutation.

## RFC and Current Status

Primary RFC:
1. `backends/qualcomm/utils/fx_viewer/RFC_FX_VIEWER_API_INTERFACE.md`

Implementation status:
1. `backends/qualcomm/utils/fx_viewer/RFC_API_IMPLEMENTATION_STATUS.md`

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

## JS API (Runtime)

Core:
1. `FXGraphViewer.create(config)`
2. `getState`, `setState`, `replaceState`, `batch`
3. `on`, `off`

Actions:
1. `setTheme`, `setLayers`, `setColorBy`
2. `selectNode`, `clearSelection`, `search`
3. `zoomToFit`, `panToNode`, `animateToNode`
4. `setUIVisibility`, `setLayout`
5. `enterFullscreen`, `exitFullscreen`

Runtime layer mutation:
1. `upsertLayer`, `removeLayer`, `patchLayerNodes`, `setLayerLabel`, `setColorRule`

Compare:
1. `FXGraphCompare.create(...)`
2. `setColumns`, `setCompact`, `setSync`, `destroy`

## Unified API Harness

Generator:
1. `backends/qualcomm/utils/fx_viewer/examples/generate_api_test_harness.py`

Template + testcase catalog:
1. `backends/qualcomm/utils/fx_viewer/examples/harness_template.html`
2. `backends/qualcomm/utils/fx_viewer/examples/harness_testcases.py`

Generated outputs:
1. `fx_viewer_api_test_harness_portable.html`
2. `fx_viewer_api_test_harness_qualcomm.html`

Testcase reference:
1. `backends/qualcomm/utils/fx_viewer/examples/FX_VIEWER_API_TESTCASES.md`

## Recommended Run (bash)

```bash
source /home/boyucwsl/executorch/.venv/bin/activate
source /home/boyucwsl/executorch/qairt/2.37.0.250724/bin/envsetup.sh
export PYTHONPATH=~/:$PYTHONPATH
python backends/qualcomm/utils/fx_viewer/examples/generate_api_test_harness.py
```

## JS Internals

See:
1. `backends/qualcomm/utils/fx_viewer/templates/README.md`
