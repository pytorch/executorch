# Observatory

Observatory is a unified debugging framework for ExecuTorch that captures graph snapshots and analysis data across compilation stages, then exports the results as a standalone, shareable HTML report.

Instead of collecting logs, traces, and artifacts from scattered sources, Observatory provides a single workflow: **capture, store, analyze, visualize, share**. The output is one HTML file that anyone can open in a browser to inspect graphs, accuracy metrics, per-layer analysis, and more.

## Why it exists

Debugging model compilation issues is often too manual. When something goes wrong, engineers typically need to collect information from multiple places, reconstruct execution context by hand, and pass partial artifacts between people. This is especially painful when the issue is hard to reproduce, the investigator is not the original developer, or the context needs to be shared across teams.

Observatory closes this gap by providing a consistent, automated workflow from data capture to presentation.

## The workflow

```
capture  -->  store  -->  analyze  -->  visualize  -->  share
```

1. **Capture**: Observatory wraps your export script and automatically collects graph snapshots at each compilation stage (export, quantize, lower).
2. **Store**: Raw data is persisted as structured JSON for later re-analysis.
3. **Analyze**: Each lens processes the collected data into findings, comparisons, and derived insights.
4. **Visualize**: Results are assembled into an interactive HTML report with multiple view types.
5. **Share**: The report is a single self-contained HTML file. Send it, attach it to a bug report, or host it on GitHub Pages.

## What you get

A standalone HTML report containing:

- **Graph View**: Interactive fx_viewer graphs with color-coded overlays (accuracy error, op type, etc.)
- **Table View**: Key-value summaries, per-record metrics, cross-record comparisons
- **Compare View**: Side-by-side graph comparison with synchronized selection
- **Dashboard**: Session-level summary with badges and navigation

## Quick start

### CLI (zero-config)

Point the CLI at any ExecuTorch export script:
The simplest invocation: point the CLI at your script and pass its arguments through.
Use `--report-html` to set output paths explicitly:
```bash
python -m devtools.observatory.cli \
    --report-html /path/to/output_report.html \
    {your original script and arguments}
```
For example:

```bash
python -m devtools.observatory.cli \
    --report-html /tmp/obs/report.html \
    --report-json /tmp/obs/report.json \
    --report-title "Swin V2-T Qualcomm" \
    examples/qualcomm/oss_scripts/swin_v2_t.py \
    --model SM8650 -b ./build-android -d imagenet-mini/val -a ./swin_v2_t 
```

Use backend-specific observatory cli for additional customized lenses and hooks (for example, xnnpack backend with per-layer accuracy analysis)

```bash
python -m backends.xnnpack.debugger.observatory.cli \
    --report-html /tmp/obs/report.html \
    --accuracy \
    examples/xnnpack/aot_compiler.py \
    --model_name=mv2 --delegate --quantize --output_dir /tmp/mv2
```

This produces:
- `/tmp/mv2/observatory_report.html` (interactive report)
- `/tmp/mv2/observatory_report.json` (raw data)

### Python API

```python
from executorch.devtools.observatory import Observatory, observe_pass

# Wrap passes for automatic graph collection
pass_a = observe_pass(SomePass())

Observatory.clear()
with Observatory.enable_context():
    # Auto: Lenses can auto-insert collection points by monkey patching when entering context
    # Manual: Insert the collection point anywhere
    Observatory.collect("step_0", graph_module)

    # observe_pass: auto-collects input and output graphs
    result = pass_a(graph_module)
    # collects "SomePass/input" and "SomePass/output"

Observatory.export_html_report("/tmp/report.html")
```

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

1. **Step 1**: Run your script, collect data, export JSON (`--json-only`)
2. **Step 2**: Convert JSON to HTML any time later (`cli visualize`)

This means you can collect data in CI and generate reports locally, or regenerate HTML after updating lens code without re-running expensive export scripts.

```bash
# Step 1: collect (e.g., in CI)
python -m devtools.observatory.cli --json-only script.py ...

# Step 2: visualize (e.g., locally)
python -m devtools.observatory.cli visualize \
    --input report.json --output report.html
```

### Fx-Viewer

Fx-Viewer (`devtools/utils/fx_viewer`) is the graph visualization component used inside Observatory's Graph View. Observatory owns the workflow; Fx-Viewer provides the interactive graph rendering, node inspection, and highlighting within that workflow.

## How to use it

See [USAGE.md](USAGE.md) for the full CLI usage guide, including:

- Zero-config e2e workflow
- JSON-only export for CI
- Visualize mode (JSON to HTML)
- Disabling lenses
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
| [lenses/DEBUG_HANDLE_SYNC_ANALYSIS.md](lenses/DEBUG_HANDLE_SYNC_ANALYSIS.md) | Technical analysis of debug_handle consistency across pipeline stages |

## Tests

```bash
pytest -q backends/devtools/observatory/tests
```
