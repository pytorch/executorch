# Observatory

Observatory is a unified debugging framework for ExecuTorch that captures graph snapshots and analysis data across compilation stages, then exports the results as a standalone, shareable HTML report.

Instead of collecting logs, traces, and artifacts from scattered sources, Observatory provides a single workflow: **capture, store, analyze, visualize, share**. The output is one HTML file that anyone can open in a browser to inspect graphs, accuracy metrics, per-layer analysis, and more.

## Why it exists

Debugging model compilation issues is often too manual. When something goes wrong, engineers typically need to collect information from multiple places, reconstruct execution context by hand, and pass partial artifacts between people. This is especially painful when the issue is hard to reproduce, the investigator is not the original developer, or the context needs to be shared across teams.

Observatory closes this gap by providing a consistent, automated workflow from data capture to presentation.

## Vocabulary (RFC §4.4-4.5)

Five nouns describe everything Observatory does:

| Term | What it is |
|------|-----------|
| **Region** | A lightweight named scope opened by `Observatory.enter_context(region_name, config=None)`. Regions can nest. Records produced inside a Region are tagged with that name in their `region_stack`. **No lens hooks fire at Region boundaries** — Regions are pure labelling for grouping records in the UI tree view. |
| **Session** | A heavyweight scope. A Session begins when an *outermost* `enter_context` is entered and ends when that block exits. Lens `on_session_start` / `on_session_end` fire here. The Session's `name` equals the outermost `region_name`. |
| **Record** | One item from `Observatory.collect(name, artifact)`. Carries `name`, `timestamp`, `session_id`, `region_stack` (snapshot at collect time), and lens-specific `digests`. Stored time-ordered. |
| **Archive** | One Observatory invocation's full state: a flat list of one or more Sessions plus every Record produced under them. The only thing Observatory persists raw. Serializes to a single JSON file via `--output-archive`. The archive label (set by `--archive`) becomes `Session.archive` and drives compare-mode column grouping. |
| **Report** | The derived output. `analyze` runs once per archive, then `Frontend.dashboard` / `Frontend.record` render Report (HTML) for human reviewers and Report (JSON) for LLMs / CI / dashboards. |

`enter_context(region_name=None, config=None)` is the primary entry API. Outermost calls open a Session named after the Region (or auto-named `default` / `default-2` ... if no `region_name` is given). Inner calls without a `region_name` are **config-only overrides** — they do not push a Region or open a Session.

## The workflow

```
capture  -->  store  -->  analyze  -->  visualize  -->  share
   ↓             ↓            ↓             ↓           ↓
 collect     Archive      analyze /     Report       attach to
 (per       (sessions[]   Frontend      (HTML +      issue / PR /
  Record)    + records[]) hooks         JSON)        email
```

1. **Capture**: Observatory wraps your export script. Built-in lenses (e.g. `pipeline_graph_collector`) install monkey-patches that call `collect(...)` at each compilation stage; you can also call `Observatory.collect(...)` directly anywhere in your code.
2. **Store**: Records + per-Session metadata are persisted as a single Archive (JSON) for later re-analysis or comparison.
3. **Analyze**: Each lens processes the Archive into findings, comparisons, and derived insights.
4. **Visualize**: Results are assembled into an interactive HTML report (Report (HTML)) with multiple view types. Use `--output-report-json` to also emit a Report (JSON) — a lens-summarised dict suitable for CI threshold checks, LLM-driven triage, and dashboard time-series ingestion (see [USAGE.md §4a](USAGE.md)).
5. **Share**: The Report is a single self-contained HTML file. Send it, attach it to a bug report, or host it on GitHub Pages.

## What you get

A standalone HTML report containing:

- **Graph View**: Interactive fx_viewer graphs with color-coded overlays (accuracy error, op type, etc.)
- **Table View**: Key-value summaries, per-record metrics, cross-record comparisons
- **Compare View**: Side-by-side graph comparison with synchronized selection
- **Session Dashboard**: Per-Session summary with badges and navigation
- **Tree-view toggle (left panel)**: Switch between flat time-ordered records and a collapsible tree grouped by `region_stack` (e.g. AOT-stage groups from `pipeline_graph_collector`)

## Quick start

### CLI (zero-config)

Point the CLI at any ExecuTorch export script:
```bash
python -m executorch.devtools.observatory SCRIPT [SCRIPT_ARGS...]
```
Use `--output-html` / `--output-archive` to set output paths explicitly:

```bash
python -m executorch.devtools.observatory \
    --output-html /tmp/obs/report.html \
    --output-archive /tmp/obs/report.json \
    examples/qualcomm/oss_scripts/swin_v2_t.py \
    --model SM8650 -b ./build-android -d imagenet-mini/val -a ./swin_v2_t
```

> Use `--archive LABEL` to name this Archive (drives compare-mode column grouping and -- when no inner `enter_context(region_name=...)` is opened by the script -- also names the default session).

Use a backend-specific observatory CLI for additional customised lenses and hooks (for example, xnnpack with per-layer accuracy):

```bash
python -m executorch.backends.xnnpack.debugger.observatory \
    --output-html /tmp/obs/report.html \
    --lens-recipe accuracy \
    examples/xnnpack/aot_compiler.py \
    --model_name=mv2 --delegate --quantize --output_dir /tmp/mv2
```

> **XNNPack note**: `aot_compiler.py` uses relative imports so it must run as a Python module.
> The CLI auto-detects this from `__init__.py` presence. You can also pass the dotted module
> name directly: `examples.xnnpack.aot_compiler`

This produces:
- `/tmp/obs/report.html` (interactive Report (HTML))
- `/tmp/obs/report.json` (Archive (JSON), path auto-derived from HTML path)

### Compare archives across backends

Overlay two Archive (JSON) files into one Report (HTML). Each archive's records and sessions are prefixed with the corresponding `--label` so identically-named pipeline stages stay distinct:

```bash
python -m executorch.devtools.observatory compare \
    --input-archive xnnpack/mv2/observatory_report.json --label XNNPACK/mv2 \
    --input-archive qualcomm/mobilenet_v2/observatory_report.json --label Qualcomm/mobilenet_v2 \
    --output-html cross_backend.html \
    --title "MobileNetV2 — XNNPACK vs Qualcomm"
```

In the resulting Report, toggle the **🌳 Tree** view in the left panel to see one collapsible region per archive, then `Select` one record from each tree and click `Compare` for a side-by-side graph diff.

### Python API

```python
from executorch.devtools.observatory import Observatory, observe_pass

# Wrap passes for automatic graph collection
pass_a = observe_pass(SomePass())

Observatory.clear()
with Observatory.enter_context("debug_run"):       # opens Session "debug_run"
    # Auto: lenses can auto-insert collection points by monkey-patching when entering context
    # Manual: insert the collection point anywhere
    Observatory.collect("step_0", graph_module)

    with Observatory.enter_context("quantize"):    # nested Region (no new Session)
        Observatory.collect("after_prepare", graph_module)
        # observe_pass: auto-collects input and output graphs
        result = pass_a(graph_module)
        # collects "SomePass/input" and "SomePass/output"

Observatory.export_html_report("/tmp/report.html")
Observatory.export_json("/tmp/report.json")        # also `Observatory.export_archive`
```

> `Observatory.enable_context(...)` (no `region_name`) is preserved as a thin alias of `enter_context()` — it auto-opens a `default` Session at the outermost call and acts as a config-only override at inner levels.

## Core concepts

### `observe_pass` decorator

The `observe_pass` decorator wraps any pass (PassBase subclass or callable) to automatically collect graphs via Observatory. By default it captures both input and output graphs, making pass debugging a one-line change:

```python
observed = observe_pass(SomePass())  # wrap once
result = observed(graph_module)       # auto-collects input + output
```

Record names are derived from the class name and auto-deduplicated on repeat calls.
See [USAGE.md](USAGE.md) for the full decorator reference.

### Lenses

A **Lens** is a modular extension that adds domain-specific debugging logic. Each lens can participate in capture, analysis, and visualization. This is what makes Observatory a framework rather than a fixed tool.

Built-in lenses:

| Lens | What it does |
|------|-------------|
| `GraphLens` | Renders interactive fx_viewer graph for each collected artifact |
| `MetadataLens` | Collects artifact type, node count, environment info |
| `StackTraceLens` | Captures the call stack at each collection point |
| `PipelineGraphCollectorLens` | Auto-collects graphs at export/quantize/lower stages (patches framework functions) |
| `AccuracyLens` | Evaluates model accuracy at each stage (PSNR, cosine similarity, MSE, top-k) |
| `GraphColorLens` | Colors graph nodes by op_type or op_target |
| `PerLayerAccuracyLens` | Computes per-layer accuracy metrics with graph overlays |

See [lenses/LENSES.md](lenses/LENSES.md) for detailed lens documentation.

### The 2-step design

Observatory separates **runtime collection** from **report generation**:

1. **Step 1**: Run your script — both JSON and HTML are exported automatically
2. **Step 2**: Re-generate HTML any time later from the JSON (`cli visualize`)

This means you can collect data in CI and re-generate reports locally, or regenerate HTML after updating lens code without re-running expensive export scripts.

```bash
# Step 1: collect (e.g., in CI)
python -m executorch.devtools.observatory script.py ...

# Step 2: re-visualize (e.g., locally)
python -m executorch.devtools.observatory visualize \
    --input-archive observatory_report.json --output-html report.html
```

### Fx-Viewer

Fx-Viewer (`devtools/utils/fx_viewer`) is the graph visualization component used inside Observatory's Graph View. Observatory owns the workflow; Fx-Viewer provides the interactive graph rendering, node inspection, and highlighting within that workflow.

## How to use it

See [USAGE.md](USAGE.md) for the full CLI usage guide, including:

- Zero-config e2e workflow
- Visualize mode (JSON to HTML)
- Manual collection points in arbitrary code
- Demo script batch modes

## Writing a custom Lens

A lens implements the `observe -> digest -> analyze -> frontend` lifecycle:

```python
from executorch.devtools.observatory.interfaces import (
    AnalysisResult, Frontend, TableBlock, TableRecordSpec, ViewList,
)

class MyLens:
    @classmethod
    def get_name(cls): return "my_lens"

    @classmethod
    def observe(cls, artifact, context):
        return {"node_count": len(artifact.graph.nodes)}

    @classmethod
    def digest(cls, observation, context):
        return observation

    @staticmethod
    def analyze(records, config):
        return AnalysisResult()

    @staticmethod
    def get_frontend_spec():
        class MyFrontend(Frontend):
            def record(self, digest, analysis, context):
                return ViewList(blocks=[
                    TableBlock(id="summary", title="Summary",
                               record=TableRecordSpec(data=digest), order=0)
                ])
        return MyFrontend()
```

Register it before entering the context:

```python
Observatory.register_lens(MyLens)
```

See [lenses/LENSES.md](lenses/LENSES.md) for the full lens protocol and built-in lens details.

## Adding graph overlays from a Lens

Lenses can contribute colored graph overlays during the `analyze()` phase:

```python
from executorch.backends.qualcomm.debugger.observatory.interfaces import (
    AnalysisResult, RecordAnalysis,
)
from executorch.devtools.fx_viewer import (
    GraphExtensionPayload, GraphExtensionNodePayload,
)

payload = GraphExtensionPayload(
    id="error", name="Accuracy Error",
    legend=[{"label": "Low", "color": "#93c5fd"}],
    nodes={"node_0": GraphExtensionNodePayload(fill_color="#93c5fd")},
)

record_analysis = RecordAnalysis(data={"max_mse": 0.1})
record_analysis.add_graph_layer("error", payload)

return AnalysisResult(per_record_data={"step_1": record_analysis})
```

## Entry points

| File | Purpose |
|------|---------|
| `observatory.py` | Runtime lifecycle, report assembly, export APIs |
| `interfaces.py` | Typed dataclass contracts for all blocks, lenses, and analysis |
| `graph_hub.py` | Graph asset/layer merge logic |
| `cli.py` | CLI runner (run mode + visualize mode) |
| `auto_collect.py` | ETRecord monkey-patch auto-collection |

## Document map

| Document | What it covers |
|----------|---------------|
| [USAGE.md](USAGE.md) | CLI usage guide, workflow examples, demo script modes |
| [lenses/LENSES.md](lenses/LENSES.md) | Built-in lens details, accuracy lens internals, custom lens patterns |
| [REFERENCE.md](REFERENCE.md) | Contract tables, API reference, JS callbacks, performance notes |

## Tests

```bash
pytest -q backends/devtools/observatory/tests
```
