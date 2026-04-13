# fx_viewer Python API Tutorial

This tutorial is intentionally practical and maps directly to harness usage.

## 1) Minimal Export

```python
import torch
from executorch.backends.qualcomm.utils.fx_viewer import FXGraphExporter

model = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU()).eval()
sample = (torch.randn(1, 16),)

ep = torch.export.export(model, sample, strict=False)
exporter = FXGraphExporter(ep.graph_module)
exporter.export_html("minimal_graph.html")
```

What you learn:
1. Create exporter from `graph_module`.
2. Generate standalone HTML quickly.

## 2) Add One Extension Layer

```python
from executorch.backends.qualcomm.utils.fx_viewer import GraphExtension

ext = GraphExtension(id="backend", name="Backend")

payload = exporter.generate_json_payload()
for node in payload["base"]["nodes"]:
    # Example: fake backend assignment
    ext.add_node_data(node["id"], {"backend": "cpu"})

ext.set_label_formatter(lambda d: [f"backend={d.get('backend', 'unknown')}"])
exporter.add_extension(ext)
exporter.export_html("graph_with_backend_layer.html")
```

What you learn:
1. `add_node_data(node_id, data)` is the core extension contract.
2. Formatter input is exactly stored extension data.

## 3) Add Color Rules

### Categorical

```python
from executorch.backends.qualcomm.utils.fx_viewer import CategoricalColorRule

ext.set_color_rule(CategoricalColorRule(attribute="backend"))
```

Use when values are discrete labels.

### Numeric

```python
from executorch.backends.qualcomm.utils.fx_viewer import NumericColorRule

metric_ext = GraphExtension(id="latency", name="Latency")
metric_ext.add_node_data("node_a", {"latency_ms": 1.2})
metric_ext.set_color_rule(NumericColorRule(attribute="latency_ms", cmap="viridis"))
```

Use when values are continuous metrics.

## 4) Export Modes

```python
payload = exporter.generate_json_payload()  # in-memory dict
exporter.export_json("graph_payload.json")
js_snippet = exporter.export_js("graph-host")
exporter.export_html("graph_standalone.html")
```

Use cases:
1. `export_html`: easiest for local inspection.
2. `export_json` + JS runtime: best for custom host applications.
3. `export_js`: quick embed in existing HTML.

## 5) debug_handle Extraction and Compare Sync

`debug_handle` is a per-node integer assigned by `generate_missing_debug_handles`. The exporter
extracts it explicitly so it is always present in `node.info` regardless of type:

```python
from executorch.exir.passes.debug_handle_generator_pass import generate_missing_debug_handles

ep = torch.export.export(model, sample, strict=False)
generate_missing_debug_handles(ep)
gm = ep.module()

exporter = FXGraphExporter(gm)
payload = exporter.generate_json_payload()
# payload["base"]["nodes"][i]["info"]["debug_handle"] is now int or list[int]
```

Fused nodes may carry a tuple handle `(h1, h2)`. The exporter normalizes:
- `int` → stored as `int`
- `tuple/list` with one element → stored as `int`
- `tuple/list` with multiple elements → stored as `list[int]`

### Registering a sync key for compare mode

To expose an extension field as an explicit sync option in the compare sidebar:

```python
ext = GraphExtension(id="my_ext", name="My Extension")
ext.add_node_data(node_id, {"debug_handle": 42, "latency_ms": 1.5})
ext.set_sync_key("debug_handle")   # appears as "Ext: my_ext.debug_handle" in sidebar
```

The `per_layer_accuracy` extension (built by `_add_accuracy_extension`) automatically registers
`debug_handle` as a sync key. This enables the compare sidebar to offer
`Ext: per_layer_accuracy.debug_handle` as an explicit sync option alongside the default
`Auto (handle→id)` mode.

## 6) Connect Python Output to JS Harness Thinking

If Python emits:
1. `extensions["per_layer_accuracy"]` (with `set_sync_key("debug_handle")`)
2. `extensions["topological_order"]`

Then JS harness can immediately use:
1. `viewer.setLayers(["per_layer_accuracy", "topological_order"])`
2. `viewer.setColorBy("per_layer_accuracy")`
3. `viewer.patchLayerNodes("per_layer_accuracy", patchByNodeId)`
4. `FXGraphCompare.create({ viewers, layout, sync: { mode: 'auto' } })` — auto sync via `debug_handle`

This is the core Python/JS contract boundary.

## 7) Recommended Practice Path

1. Start with `minimal_graph.html`.
2. Add one extension with one field.
3. Add categorical color.
4. Add numeric metric layer.
5. Add `set_sync_key` and test in compare mode (`js_08`, `adv_04`).
6. Run `demo_3graph_compare.py` to see all three `debug_handle` mapping patterns.
