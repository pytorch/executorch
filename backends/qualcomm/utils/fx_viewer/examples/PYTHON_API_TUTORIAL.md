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

## 5) Connect Python Output to JS Harness Thinking

If Python emits:
1. `extensions["per_layer_accuracy"]`
2. `extensions["topological_order"]`

Then JS harness can immediately use:
1. `viewer.setLayers(["per_layer_accuracy", "topological_order"])`
2. `viewer.setColorBy("per_layer_accuracy")`
3. `viewer.patchLayerNodes("per_layer_accuracy", patchByNodeId)`

This is the core Python/JS contract boundary.

## 6) Recommended Practice Path

1. Start with `minimal_graph.html`.
2. Add one extension with one field.
3. Add categorical color.
4. Add numeric metric layer.
5. Validate behavior in JS harness beginner cases (`js_04`, `js_05`).
