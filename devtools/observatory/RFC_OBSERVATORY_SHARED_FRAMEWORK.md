# RFC: Observatory & fx_viewer — Shared Debugging Framework for ExecuTorch

**Date:** 2026-04-13
**Status:** Draft for review
**Authors:** Qualcomm Innovation Center, Inc.
**Scope:** Promote Observatory and fx_viewer from `backends/qualcomm/` to `devtools/` as shared ExecuTorch debugging infrastructure

---

## 1. Abstract

Observatory is a unified, extensible debugging framework for ExecuTorch that turns scattered debugging artifacts into structured, reviewable packages. It supports the full workflow from data collection and storage to analysis and visualization, exporting results as standalone shareable HTML reports. Its extensibility comes from **Lenses** — modular extensions that contribute logic at each stage of the workflow.

fx_viewer is the interactive graph visualization component used within Observatory's Graph View for FX graph inspection, annotation, and comparison.

Both tools currently live under `backends/qualcomm/` but are architecturally generic — they work with any `torch.fx.GraphModule` and any ExecuTorch backend. This RFC proposes moving the shared framework to `devtools/` so all backends can use it, while allowing each backend to maintain its own CLI runner and custom lenses for backend-specific debugging needs.

## 2. Background

Debugging model execution in ExecuTorch today involves multiple disconnected tools and manual workflows:

- **ExecuTorch infrastructure** (Inspector API, ETRecord, ETDump) provides raw materials — intermediate graphs, runtime artifacts, operator tracking, profiling data — but analysis and presentation remain minimal. Developers write custom scripts to answer practical questions.

- **Existing visualizers** (QAIRT Visualizer, Model Explorer) offer graph viewing and table inspection, but none natively support `torch.fx` graphs, and most require manual export preparation and custom configuration.

- **Specialized debuggers** target narrow use cases (e.g., per-layer accuracy tracking) but require complicated setup and don't scale to large models or diverse debugging scenarios.

The missing piece is **workflow glue**: a practical way to capture many kinds of debugging context, store them coherently, analyze them consistently, and export the result into a form others can explore interactively.

## 3. Problem Statement

1. **Fragmented debugging workflow.** When something goes wrong, engineers collect information from different places, reconstruct execution context by hand, and pass around partial artifacts. This is especially painful when the issue is hard to reproduce, the investigator is not the original developer, or context must be shared across teams.

2. **No shared debugging framework across backends.** Each ExecuTorch backend (Qualcomm, XNNPACK, ARM, Vulkan, etc.) faces similar debugging challenges — accuracy regression, graph transformation issues, performance analysis — but there is no shared infrastructure for building debugging tools. Each backend reinvents the wheel.

3. **No native FX graph visualization.** Despite `torch.fx` being the core IR for ExecuTorch compilation, no existing tool provides interactive visualization of FX graphs with annotation layers, comparison views, and integration into a broader debugging workflow.

4. **Debugging artifacts are not shareable by default.** Captured data feels like raw leftovers from a failed run rather than a useful debugging package that others can open, reason about, and act on.

## 4. Goals

1. **Shared framework in `devtools/`** — Observatory core, fx_viewer, and generic lenses available to all backends via `executorch.devtools.observatory` and `executorch.devtools.fx_viewer`.

2. **Backend-extensible** — Each backend can implement custom lenses and maintain its own CLI runner that enables a specific set of lenses for its debugging needs.

3. **Unified lens composition** — Users can mix shared lenses (from `devtools/`) and backend-specific lenses (from `backends/<name>/`) in a single Observatory session.

4. **End-to-end workflow** — Support the full debugging lifecycle: capture → store → analyze → visualize → share → diagnose.

5. **Standalone shareable reports** — Export self-contained HTML reports with multiple view types (Table, Custom HTML, Graph) that anyone can open in a browser.

6. **Foundation for automation** — Structured debugging artifacts that can serve as inputs for automated analysis and AI-assisted triage.

## 5. Non-Goals

- Replacing existing ExecuTorch infrastructure (Inspector API, ETRecord, ETDump). Observatory complements these tools and can consume their outputs.
- Replacing external visualizers (Model Explorer, QAIRT Visualizer). fx_viewer focuses specifically on FX graph visualization within the Observatory workflow.
- Mandating that all backends adopt Observatory. The framework is opt-in; backends adopt it when they have debugging needs it can serve.
- Building a real-time debugging UI. Observatory produces post-hoc reports, not live dashboards.

## 6. Proposal

### 6.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    devtools/observatory/                 │
│                                                         │
│  Observatory Runtime    Lens Protocol    Report Engine   │
│  ┌──────────────┐    ┌─────────────┐   ┌────────────┐  │
│  │ collect()    │    │ observe()   │   │ HTML/JSON  │  │
│  │ analyze()    │◄──►│ digest()    │   │ export     │  │
│  │ export()     │    │ analyze()   │   │            │  │
│  └──────────────┘    │ frontend()  │   └────────────┘  │
│                      └─────────────┘                    │
│  Generic Lenses: Graph, Metadata, Accuracy, StackTrace  │
│                                                         │
│  devtools/fx_viewer/                                    │
│  ┌──────────────────────────────────────────────────┐   │
│  │ FXGraphExporter  GraphExtension  ColorRules      │   │
│  │ JS Runtime (canvas, minimap, search, compare)    │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
         ▲                    ▲                    ▲
         │                    │                    │
┌────────┴───┐    ┌──────────┴──────┐    ┌───────┴────────┐
│ QNN CLI    │    │ XNNPACK CLI     │    │ ARM CLI        │
│ QNN Lenses │    │ XNNPACK Lenses  │    │ ARM Lenses     │
│ backends/  │    │ backends/       │    │ backends/      │
│ qualcomm/  │    │ xnnpack/        │    │ arm/           │
└────────────┘    └─────────────────┘    └────────────────┘
```

### 6.2 The Lens Protocol

A Lens is a modular extension that adds domain-specific debugging logic to Observatory. A Lens participates in one or more stages of the workflow:

```python
class Lens:
    @classmethod
    def setup(cls) -> None:
        """One-time initialization."""

    @classmethod
    def observe(cls, artifact, context) -> Any:
        """Capture: intercept artifacts during script execution."""

    @classmethod
    def digest(cls, observation, context) -> Serializable:
        """Store: convert observation to JSON-serializable form."""

    @staticmethod
    def analyze(records, config) -> AnalysisResult:
        """Analyze: compute derived data from collected records."""

    @staticmethod
    def get_frontend_spec() -> Frontend:
        """Visualize: define how results are presented in the report."""
```

**Why Lenses matter:** The debugging needs of ExecuTorch development are not uniform. Different users care about different questions — accuracy regression, performance, graph transformation correctness, runtime behavior. A single hardcoded tool cannot serve all of these well. Lenses make it possible to keep the overall workflow consistent while allowing the debugging logic to remain specialized and customizable.

### 6.3 Frontend Block Types

Observatory reports support four block types, each with record and compare semantics:

| Block Type | Purpose | Compare Mode |
|---|---|---|
| **TableBlock** | Key-value data tables | Auto side-by-side diff |
| **HtmlBlock** | Raw HTML fragments | Auto side-by-side |
| **CustomBlock** | Custom JS rendering | Custom JS or auto |
| **GraphBlock** | Interactive FX graph viewer | Synchronized N-way compare |

### 6.4 fx_viewer: Graph Visualization

fx_viewer provides:

- **Python export layer**: Normalizes FX metadata, computes hierarchical layout (Sugiyama algorithm), generates extensible JSON payloads
- **Extension system** (`GraphExtension`): Overlays custom annotation layers with formatters and color rules
- **Browser runtime**: Interactive canvas with pan/zoom, minimap, fuzzy search, multi-layer coloring, state-driven API
- **Compare mode**: N-way synchronized graph views with merged info panels and debug_handle-aware sync

### 6.5 Backend-Specific Extensions

Each backend can extend Observatory in two ways:

**1. Backend patch registration** — Hook into shared lenses to add backend-specific compile-stage interception:

```python
# backends/qualcomm/debugger/observatory/cli.py
from executorch.devtools.observatory.lenses.pipeline_graph_collector import PipelineGraphCollectorLens
from .lenses.qnn_patches import install_qnn_patches

PipelineGraphCollectorLens.register_backend_patches(install_qnn_patches)
```

**2. Custom lenses** — Implement the Lens protocol for backend-specific debugging needs:

```python
# backends/arm/debugger/observatory/lenses/ethos_u_profiling.py
from executorch.devtools.observatory.interfaces import Lens

class EthosUProfilingLens(Lens):
    @classmethod
    def get_name(cls) -> str:
        return "ethos_u_profiling"
    # ... implement observe, analyze, frontend ...
```

### 6.6 Backend CLI Runners

Each backend maintains its own CLI that registers the appropriate lenses. Features are opt-in via flags — the default mode is graph collection only.

```bash
# Generic (framework lenses only, graph collection)
python -m executorch.devtools.observatory SCRIPT [ARGS...]

# Qualcomm — graph collection only (default)
python -m executorch.backends.qualcomm.debugger.observatory SCRIPT [ARGS...]

# Qualcomm — with accuracy debugging
python -m executorch.backends.qualcomm.debugger.observatory --lense_recipe=accuracy SCRIPT [ARGS...]

# XNNPACK — graph collection only (default)
python -m executorch.backends.xnnpack.debugger.observatory SCRIPT [ARGS...]

# XNNPACK — with accuracy debugging
python -m executorch.backends.xnnpack.debugger.observatory --lense_recipe=accuracy SCRIPT [ARGS...]
```

### 6.7 fx_viewer Extension API

fx_viewer's Python API allows lenses and backend tools to customize graph visualization through `GraphExtension`. Each extension becomes a toggleable layer in the viewer with its own data, coloring, labeling, and sync rules. This is how Observatory lenses contribute graph overlays.

**Example context: Per-Layer Accuracy Lens** — A lens that computes per-operator accuracy metrics (PSNR, cosine similarity) and overlays them on the graph as a color-coded layer.

#### 6.7.1 Custom Layer Data (Info Panel)

Each extension can attach arbitrary key-value data to nodes. When a node is selected, extension data appears in the viewer's info panel, prefixed by the extension name.

```python
from executorch.devtools.fx_viewer.extension import GraphExtension

# Create an extension layer
accuracy_ext = GraphExtension(id="per_layer_accuracy", name="Per-Layer Accuracy")

# Attach per-node metrics — these appear in the info panel when the node is selected
for node_id, metrics in per_node_results.items():
    accuracy_ext.add_node_data(node_id, {
        "psnr_db": f"{metrics['psnr']:.2f}",
        "cosine_similarity": f"{metrics['cosine']:.4f}",
        "mse": f"{metrics['mse']:.6f}",
        "sample_index": metrics["worst_sample_idx"],
    })
```

In the viewer, selecting a node shows:
```
── Base ──
op: call_function
target: aten::conv2d.default
tensor_shape: [1, 64, 224, 224]

── Per-Layer Accuracy ──
psnr_db: 42.31
cosine_similarity: 0.9987
mse: 0.000012
sample_index: 7
```

#### 6.7.2 Custom Node Labeling and Tooltips

Extensions can append text to node labels and tooltips. Label formatters receive the node's extension data and return lines to append below the base label. Tooltip formatters work similarly for hover text.

```python
# Custom label: show PSNR value below the node name
accuracy_ext.set_label_formatter(
    lambda node_data: [f"PSNR: {node_data.get('psnr_db', 'N/A')}"]
)

# Custom tooltip: show detailed metrics on hover
accuracy_ext.set_tooltip_formatter(
    lambda node_data: [
        f"PSNR: {node_data.get('psnr_db', 'N/A')} dB",
        f"Cosine: {node_data.get('cosine_similarity', 'N/A')}",
        f"Worst sample: #{node_data.get('sample_index', '?')}",
    ]
)
```

In the viewer, a node renders as:
```
┌──────────────────┐
│    conv2d         │  ← base label
│  PSNR: 42.31     │  ← extension label_append
└──────────────────┘
```

#### 6.7.3 Custom Coloring

Extensions define how nodes are colored using `ColorRule` subclasses. Two built-in rules cover most cases:

- **`NumericColorRule`**: Maps a continuous metric to a color gradient (viridis, reds, blues, greens). Ideal for accuracy metrics.
- **`CategoricalColorRule`**: Maps discrete values to deterministic hues (MD5 hash → HSV). Ideal for op types, backend assignments, etc.

```python
from executorch.devtools.fx_viewer.color_rules import NumericColorRule, CategoricalColorRule

# Color nodes by PSNR: low PSNR (red) → high PSNR (green)
# Nodes with low accuracy stand out visually
accuracy_ext.set_color_rule(
    NumericColorRule(
        field="psnr_db",
        palette="reds_reversed",  # low values = dark red, high = light
        label="PSNR (dB)",
    )
)

# Alternative: color by a categorical field
backend_ext = GraphExtension(id="backend", name="Backend Assignment")
backend_ext.set_color_rule(
    CategoricalColorRule(field="backend")  # auto-assigns colors: "cpu" → blue, "qnn" → orange, etc.
)
```

When the user selects "Per-Layer Accuracy" as the active color-by layer in the viewer, all nodes are recolored by their PSNR values with a gradient legend.

#### 6.7.4 Custom Sync Rules (Compare Mode)

In compare mode (N-way synchronized graph views), extensions can define sync keys — fields used to match corresponding nodes across different graphs. This is critical when graphs have different node IDs after transformations (fusion, decomposition).

```python
# Register a sync key so compare mode can match nodes across graphs
# by their accuracy metric values (useful when node IDs differ after fusion)
accuracy_ext.set_sync_key("from_node")  # match by original source node reference
```

When a user clicks a node in Graph A, the compare view uses the sync key to find the corresponding node in Graph B and auto-selects it, even if the node IDs differ.

#### 6.7.5 Putting It All Together

A complete example showing how a lens builds an fx_viewer extension during the analyze phase:

```python
from executorch.devtools.fx_viewer.extension import GraphExtension
from executorch.devtools.fx_viewer.color_rules import NumericColorRule
from executorch.devtools.observatory.interfaces import AnalysisResult, GraphLayerContribution

class PerLayerAccuracyLens(Lens):
    @staticmethod
    def analyze(records, config) -> AnalysisResult:
        result = AnalysisResult()

        for record_name, record in records.items():
            # Build the extension layer
            ext = GraphExtension(id="per_layer_accuracy", name="Per-Layer Accuracy")

            # 1. Attach per-node data (info panel)
            for node_id, metrics in compute_metrics(record).items():
                ext.add_node_data(node_id, {
                    "psnr_db": f"{metrics['psnr']:.2f}",
                    "cosine": f"{metrics['cosine']:.4f}",
                })

            # 2. Custom labeling
            ext.set_label_formatter(lambda d: [f"PSNR: {d.get('psnr_db', '')}"])

            # 3. Custom coloring
            ext.set_color_rule(NumericColorRule(
                field="psnr_db", palette="reds_reversed", label="PSNR (dB)"
            ))

            # 4. Custom sync rule for compare mode
            ext.set_sync_key("from_node")

            # 5. Contribute the layer to the Observatory report
            result.per_record[record_name] = RecordAnalysis(
                graph_layers=[GraphLayerContribution(extension=ext)]
            )

        return result
```

This single extension produces: colored nodes in the graph view, PSNR labels on each node, detailed metrics in the info panel, and cross-graph sync in compare mode — all from the Python API, with no JavaScript required.

## 7. Proposed Directory Structure

```
devtools/
  fx_viewer/                              # Shared graph visualization
    __init__.py, exporter.py, models.py, extension.py, color_rules.py
    grandalf/                             # Vendored layout engine
    templates/                            # JS runtime

  observatory/                            # Shared debugging framework
    __init__.py, __main__.py
    observatory.py, interfaces.py, cli.py
    graph_hub.py, html_template.py, template_loader.py, utils.py
    templates/                            # Observatory JS/CSS runtime
    lenses/
      graph.py, graph_color.py, metadata.py, stack_trace.py
      accuracy.py, per_layer_accuracy.py, pipeline_graph_collector.py

backends/<name>/debugger/observatory/     # Backend-specific extensions
    cli.py, __main__.py
    lenses/
      <backend>_patches.py
      <backend>_lenses.py
```

## 8. Breaking Changes

All imports change from `executorch.backends.qualcomm.*` to `executorch.devtools.*`:

| Before | After |
|---|---|
| `executorch.backends.qualcomm.utils.fx_viewer` | `executorch.devtools.fx_viewer` |
| `executorch.backends.qualcomm.debugger.observatory` | `executorch.devtools.observatory` |

This is a clean break — no backward-compatibility shims. All existing callers must update their imports.

## 9. Alternatives Considered

**A. Keep everything in `backends/qualcomm/` and have other backends depend on it.**
Rejected: Creates an artificial dependency on the Qualcomm backend package for generic debugging tools. Confusing for contributors and violates the principle that shared tools belong in shared locations.

**B. Move to a new top-level `tools/` or `debugging/` directory.**
Rejected: ExecuTorch already has `devtools/` as the established home for shared developer tools (Inspector, ETDump, ETRecord, visualization). Observatory and fx_viewer belong there.

**C. Use Python entry points for automatic lens discovery.**
Rejected for now: Explicit registration via `Observatory.register_lens()` is simpler, more predictable, and sufficient for current needs. Entry-point discovery can be added later without changing the core runtime.

**D. Provide backward-compatibility shims at old import paths.**
Rejected: Adds maintenance burden and delays migration. A clean break is simpler and encourages adoption of the canonical paths.

## 10. Implementation Plan

| Phase | Description |
|---|---|
| 1 | Move fx_viewer to `devtools/fx_viewer/`, delete old path, update imports |
| 2 | Move observatory core + generic lenses to `devtools/observatory/`, update imports |
| 3 | Split backend-specific patches from shared lenses (QNN → qualcomm, XNNPACK → xnnpack) |
| 4 | Create generic CLI + backend-specific CLIs for Qualcomm and XNNPACK |
| 5 | Delete old paths, update all imports across codebase, move tests |

## 11. Risk Analysis

| Risk | Mitigation |
|---|---|
| `__file__`-relative paths break after move | Templates move with code; verified by import tests |
| Cross-lens state sharing breaks | State pattern (`_worst_indices`, `_last_calibration_dataset`) is class-level and import-path independent |
| Grandalf GPL v2 license concerns | Already vendored; LICENSE file moves with code; no license change |
| Backend teams unfamiliar with Lens protocol | Provide clear documentation, examples, and reference implementations (QNN, XNNPACK) |

## 12. Test Strategy

1. Import verification for all new paths
2. Existing observatory unit tests migrated and passing from new locations
3. End-to-end report generation via each backend CLI
4. Grep verification that no old import paths remain in `devtools/`

## 13. Rollout and Adoption

1. **Initial PR**: Move fx_viewer and observatory core to `devtools/`, create QNN and XNNPACK backend CLIs
2. **Documentation**: Update devtools overview, add Observatory to the devtools landing page
3. **Community engagement**: Share RFC, demo the workflow, gather feedback
4. **Incremental adoption**: Other backends (ARM, Vulkan, etc.) can add their own CLIs and lenses as needed

## 14. Open Questions

1. Should the generic CLI (`python -m executorch.devtools.observatory`) auto-discover backend patches if the backend package is installed, or should it always require explicit backend CLI usage?
2. Should Observatory integrate with the existing Inspector API, or remain a parallel workflow?
3. ~~What is the right granularity for the `--accuracy` flag?~~ Resolved: backend CLIs use `--lense_recipe=accuracy` to opt-in to accuracy lenses. The generic CLI has no accuracy flag.

## 15. Expected Outcome

- **Faster understanding**: Engineers spend less time reconstructing context and more time solving problems
- **Better issue handoff**: Developers, QA, and reviewers work from the same debugging package
- **Better reproducibility**: Issue reports become more actionable with consistent artifact capture
- **Better collaboration**: Supports both internal engineering and external community engagement
- **Foundation for automation**: Structured debugging data enables future AI-assisted triage
- **Backend ecosystem**: Every ExecuTorch backend can build on a shared debugging framework instead of reinventing the wheel
