## What This is fx_viewer

`fx_viewer` exports a PyTorch model graph and renders it as an interactive browser viewer.

- Python side:
  - traces/extracts FX graph
  - computes layout (Grandalf/Sugiyama)
  - builds payload (`base` + `extensions`)
  - emits JSON/JS/HTML
- JavaScript side:
  - renders graph canvas + minimap
  - handles selection, search, zoom/pan
  - toggles extension layers
  - applies color-by mode
  
## Why Yet Another Graph Visualizer?

### Simplicity
- The whole visualization frontend is done within 2k lines of vallina Javascript (no library or dependency).
- No installation required, export standalone html that runs in any browser.

### Integration & Customization
- Support Python extension to customize color display, insert additional data and labels, enable easily integration with executorch debuggers and profiling utilities. 
- Simple JS API for embedding visualizer in custom HTML div, easily control or customize the interactive actions. 

### Performance
- Easily render 10k+ nodes, load the entire graph instantly.
- Much faster and smooth experience due to lightweight design and pre-computed graph layout. (compared to dagre based JS engines i.e. netron / model-explorer).




## Quick Start

Use your executorch venv:

```bash
source ~/executorch/.venv/bin/activate
```

Run extension demo (Swin + Llama):

```bash
python examples/demo_fx_viewer_extensions.py --model both
```

This generates:
- `swin_graph_v3_extensions.html`
- `llama_graph_v3_extensions.html`

## Python API

Core API:

```python
from fx_viewer import FXGraphExporter, GraphExtension, CategoricalColorRule

ep_model = torch.export.export(model, inputs, strict=False)
graph_module = ep_model.graph_module

exporter = FXGraphExporter(graph_module)

ext = GraphExtension(id="backend_type", name="Backend Assignment")
ext.add_node_data("node_id1", {"backend": "cpu"})
ext.add_node_data("node_id2", {"backend": "gpu"})
ext.set_color_rule(CategoricalColorRule(attribute="backend"))

exporter.add_extension(ext)
exporter.export_html("graph.html")
```

Export options:
- `generate_json_payload()`: return payload dict in memory
- `export_json(path)`: write payload as JSON file
- `export_js(container_id)`: return embeddable JS snippet
- `export_html(path)`: write standalone HTML

## Canonical Data Contract

The exporter now uses typed wire-format dataclasses:

- `GraphNode`:
  - `id`, `label`, `x`, `y`, `width`, `height`, `info`, `tooltip`, `fill_color`
- `GraphEdge`:
  - `v`, `w`, `points`

Formatters should consume `GraphNode` only. Required processing fields like
`name/op/target/args/kwargs` are read from `node.info`.

### JSON Schema Mapping

The payload is emitted with `dataclasses.asdict(...)`, so these dataclasses are
the JSON schema source of truth.

`GraphNode` -> `base.nodes[]`

| Field | Type | Notes |
| --- | --- | --- |
| `id` | `str` | Node id (FX name) |
| `label` | `str` | Rendered title text |
| `x`, `y` | `float` | Layout position |
| `width`, `height` | `float` | Layout box size |
| `info` | `dict[str, Any]` | Core metadata used by search/info panel |
| `tooltip` | `list[str]` | Base tooltip lines |
| `fill_color` | `str \| None` | Optional node color |

`GraphEdge` -> `base.edges[]`

| Field | Type | Notes |
| --- | --- | --- |
| `v` | `str` | Source node id |
| `w` | `str` | Target node id |
| `points` | `list[{x: float, y: float}]` | Optional routed polyline |

Top-level:
- `GraphPayload.base` -> `{legend, nodes, edges}`
- `GraphPayload.extensions` -> extension overlays keyed by extension id

## Exporter Architecture (Phases)

`FXGraphExporter.generate_json_payload()` is split into explicit phases:

1. `trace_model()`
2. `extract_graph()`
3. `compute_layout()`
4. `build_base_payload()`
5. `build_extensions_payload()`

This separation keeps behavior reviewable and testable.

## JS Architecture

The viewer is split into modules under `fx_viewer/templates/`:

- `themes.js`
- `graph_data_store.js`
- `search_engine.js`
- `view_controller.js`
- `canvas_renderer.js`
- `minimap_renderer.js`
- `ui_manager.js`
- `fx_graph_viewer.js`

Detailed JS API and load order are documented in:
- `fx_viewer/templates/README.md`

## Information Flows

Extension toggle flow:
1. UI checkbox/radio changes
2. `ViewerController.setState(...)`
3. `GraphDataStore.computeActiveGraph(...)`
4. minimap/legend/info refresh
5. full re-render

Selection flow:
1. canvas click
2. controller computes ancestors/descendants
3. canvas + minimap highlight path
4. info panel shows merged metadata

Search flow:
1. user types query
2. `SearchEngine.search(...)`
3. candidates shown in dropdown
4. hover/enter navigates or selects

## Extension Authoring Guide

`GraphExtension` adds optional node-level overlays on top of base graph structure.

### Extension Working Logic (Key Contract)

This is the most important extension contract:

1. You populate extension data explicitly with `add_node_data(node_id, data)`.
2. `label_formatter(data)` and `tooltip_formatter(data)` receive exactly that stored `data` dict.
3. Formatters must only read keys that were explicitly written previously via `add_node_data(...)`.

What formatters do **not** get automatically:
- full FX node object
- base graph node `info`
- global graph context

Return contract:
- formatter output must be `list[str]`
- invalid output (or formatter exceptions) is ignored with warnings

If you need base attributes (for example `target`, `op`) in extension label/tooltip,
copy them into extension `data` first, then read from formatter input.

### Extension Skeleton

```python
from fx_viewer import GraphExtension

ext = GraphExtension(id="my_ext", name="My Extension")

# Attach data to node ids from exported graph
ext.add_node_data("node_1", {"metric": 3.14, "tag": "hot"})

# Optional text inside node
ext.set_label_formatter(lambda data: [f"metric={data['metric']:.2f}"])

# Optional tooltip lines
ext.set_tooltip_formatter(lambda data: [f"tag={data['tag']}"])
```

### Good vs Bad Formatter Usage

```python
# GOOD: formatter reads only keys explicitly added before
ext.add_node_data(node_id, {"target": "aten.add", "latency_ms": 1.2})
ext.set_label_formatter(lambda data: [f"target={data['target']}"])
ext.set_tooltip_formatter(lambda data: [f"latency={data['latency_ms']}"])

# BAD: formatter assumes implicit fields that were never added
ext.set_label_formatter(lambda data: [f"shape={data['tensor_shape']}"])  # KeyError risk
```

Validation behavior:
- extension `id` and `name` must be non-empty
- extension IDs must be unique within one exporter
- formatters must return `list[str]`
- formatter failures emit warnings with extension/node context

### Color Rules

Color rules map extension data to `fill_color` and legend entries.

Available rules:
- `CategoricalColorRule(attribute, color_map=None)`
- `NumericColorRule(attribute, cmap="viridis", handle_outliers=True)`

#### CategoricalColorRule

Use for string-like buckets (`op type`, `device`, `stage`).

```python
from fx_viewer import CategoricalColorRule

ext.set_color_rule(CategoricalColorRule(
    attribute="op",
    color_map={"conv": "#ff9999", "linear": "#99ccff"}
))
```

Behavior:
- if `color_map` has a key, use it
- otherwise, deterministic hash-based color is generated
- legend is stable across runs for same values

Use when:
- categories are discrete
- relative ordering is not meaningful

#### NumericColorRule

Use for continuous metrics (`latency`, `memory`, `topo_index`).

```python
from fx_viewer import NumericColorRule

ext.set_color_rule(NumericColorRule(
    attribute="latency_ms",
    cmap="viridis",          # or Reds/Blues/Greens
    handle_outliers=True
))
```

Behavior:
- normalizes values into `[min, max]`
- optional percentile clipping for outliers
- generates 5-step legend

Use when:
- magnitude matters
- you want heatmap-like visual scanning

### Practical Rule Selection

- Prefer categorical when the value domain is small and semantic.
- Prefer numeric when values are measured quantities.
- For noisy metrics with extreme spikes, keep `handle_outliers=True`.
- For rank/index-like fields (`topological_order`), set `handle_outliers=False`.

## Testing

Contract tests live in:
- `tests/test_exporter_contract.py`

They validate:
- default payload shape
- custom base label/tooltip formatter behavior
- extension merge behavior
- color rule legend stability

Run:

```bash
source ~/executorch/.venv/bin/activate
pytest -q tests/test_exporter_contract.py
```
