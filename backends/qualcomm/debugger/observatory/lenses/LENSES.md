# Observatory Lenses Reference

## Overview

Observatory lenses are plugins that observe, analyze, and render model artifacts
at each stage of the ExecuTorch compilation pipeline. Lenses install their own
monkey-patches in `on_session_start()` and remove them in `on_session_end()`.

## Built-in Lenses

| Lens | Purpose | Patches? |
|------|---------|----------|
| GraphLens | Renders fx_viewer graph visualization | No |
| MetadataLens | Collects artifact type, node count, environment info | No |
| StackTraceLens | Captures repo-local call stack at collection time | No |
| PipelineGraphCollectorLens | Auto-collects graphs at each pipeline stage | Yes |
| AccuracyLens | Evaluates model accuracy at each stage | Yes |
| PerLayerAccuracyLens | Sparse per-layer accuracy via `from_node_root` matching | No |

---

## PipelineGraphCollectorLens

### Purpose

Automatically collects graph snapshots at each stage of the export -> quantize ->
lower pipeline by monkey-patching framework-level functions plus backend-specific
entrypoints. Framework-level patches cover common stages (`prepare_pt2e`,
`convert_pt2e`, `to_edge_transform_and_lower`), while backend-specific patches
are used to collect "Exported Float" with stable fallback dataset capture.

### Observation Points

| # | Pipeline Stage | Record Name | Patched Function | Source File | Collected Artifact |
|---|---------------|-------------|-----------------|-------------|-------------------|
| 1 | Backend-specific pre-quant (QNN) | "Exported Float" | `ptq_calibrate()` | `executorch/examples/qualcomm/utils.py` | `ExportedProgram` (`run_decompositions({})`) |
| 2 | Backend-specific pre-quant (XNNPACK) | "Exported Float" | `quantize()` | `executorch/examples/xnnpack/quantization/utils.py` | `ExportedProgram` (`run_decompositions({})`) |
| 3 | Quantizer Prepare | "Annotated Model" | `prepare_pt2e()` | `torchao/.../quantize_pt2e.py` | Output `GraphModule` with observers |
| 4 | Quantizer Convert (input) | "Calibrated Model" | `convert_pt2e()` | same | Input `GraphModule` (post-calibration) |
| 5 | Quantizer Convert (output) | "Quantized Model" | `convert_pt2e()` | same | Output `GraphModule` with Q/DQ ops |
| 6 | Edge transform input | "Pre-EdgeTransform/{method}" | `to_edge_transform_and_lower()` | `executorch/exir/program/_program.py` + `executorch/exir/__init__.py` | Input `ExportedProgram` (single or dict entry) |
| 7 | ETRecord Export | "ETRecord Exported/{method}" | `ETRecord.add_exported_program()` | `executorch/devtools/etrecord/` | Exported program |
| 8 | ETRecord Edge | "ETRecord Edge/{method}" | `ETRecord.add_edge_dialect_program()` | same | Edge dialect program |
| 9 | ETRecord Extra | "ETRecord Extra/{module}" | `ETRecord.add_extra_export_modules()` | same | Extra modules |
| 10 | Edge transform output (final) | "EdgeProgramManager EP" | `to_edge_transform_and_lower()` | `executorch/exir/program/_program.py` + `executorch/exir/__init__.py` | `EdgeProgramManager.exported_program()` |

### Patching Strategy

Each patch follows the same pattern:
1. Save original function in `_originals[key]`
2. Create wrapper that calls `Observatory.collect(name, artifact)` then calls original
3. Replace function in module namespace via `setattr`
4. On session end, restore all originals

Patch install order is explicit:
1. backend-agnostic framework patches (`torchao`, `executorch.exir`, ETRecord)
2. backend-specific patches (QNN/XNNPACK)

This ordering avoids early-import alias freezing in e2e scripts.

The `to_edge_transform_and_lower` patch also forces `generate_etrecord=True` to
ensure ETRecord collection fires (rows 7-9). The post-call edge output record
("EdgeProgramManager EP") is collected after ETRecord hooks complete.

### Backend Contract for AccuracyLens

`PipelineGraphCollectorLens` owns a cross-lens fallback dataset field:

- `_last_calibration_dataset`

Contract:
- Any backend-specific patch that emits `"Exported Float"` must also populate
  `_last_calibration_dataset`.
- This is done through `_set_accuracy_fallback_dataset(...)` in
  `pipeline_graph_collector.py`.
- AccuracyLens uses this field when dataset loader patches did not provide
  `_captured_dataset`.

Current backend-specific implementations:
- QNN: `ptq_calibrate` patch
- XNNPACK: `quantize` patch

---

## AccuracyLens

### Purpose

Evaluates model accuracy at each collected pipeline stage by running inference
with a dataset and computing metrics (TopK, PSNR, CosineSimilarity, etc.).

### How It Works

AccuracyLens depends on PipelineGraphCollectorLens for graph collection timing.
It configures itself lazily when it first observes the "Exported Float" record:

```
Observatory.collect("Exported Float", exported_program)
  -> AccuracyLens.observe() triggered
  -> Recognizes record name "Exported Float"
  -> Extracts float model from ExportedProgram
  -> Configures evaluator with captured dataset + auto-detected metrics
  -> Runs evaluation on float model -> returns accuracy digest
  -> Evaluator is now ready for subsequent records

Observatory.collect("Quantized Model", quantized_model)
  -> AccuracyLens.observe() triggered
  -> Evaluator already configured
  -> Runs evaluation on quantized model -> returns accuracy digest
```

### Data Sources and Fallback Strategy

| Data | Primary Source | Fallback | When Fallback Triggers |
|------|---------------|----------|----------------------|
| Dataset (inputs) | `get_imagenet_dataset` / `get_masked_language_model_dataset` patches | `PipelineGraphCollectorLens._last_calibration_dataset` from backend-specific patch | Dataset loader patches do not fire or do not capture usable inputs |
| Targets (labels) | `get_imagenet_dataset` / `get_masked_language_model_dataset` patch | None (skip target-specific metrics) | Custom dataset, non-classification task |
| Float model | "Exported Float" artifact (ExportedProgram) | -- | Always available |
| Golden outputs | Computed from float model + dataset | -- | Always available if dataset exists |
| Post-process | Auto-detected from model output type | Identity function | Detection failure |

**Fallback behavior when dataset patches don't fire:**
- AccuracyLens uses backend-captured fallback dataset from PipelineGraphCollectorLens
- Target-dependent metrics (TopK/MaskedTokenAccuracy) are skipped when targets are unavailable
- Golden-output metrics (PSNR, CosineSimilarity, MSE, AbsErr) still run when fallback inputs exist

### Dataset Loader Patches

| Patched Function | Module | Return Format | What's Captured |
|-----------------|--------|---------------|-----------------|
| `get_imagenet_dataset` | `executorch.examples.qualcomm.utils` | `(List[Tuple[Tensor]], List[Tensor])` | inputs + class targets |
| `get_masked_language_model_dataset` | `executorch.examples.qualcomm.utils` | `(List[Tuple[Tensor, Tensor]], List[Tensor])` | inputs + masked targets |

Note: `AccuracyLens` no longer patches `build_executorch_binary`; dataset fallback is
owned by `PipelineGraphCollectorLens` backend-specific contract.

### Auto-Detection

**Task type** (from target format):
- Targets contain -100 values -> MLM mode -> uses `MaskedTokenAccuracy` + `MLMEvaluator`
- Otherwise -> Classification mode -> uses `TopKAccuracy` + `StandardEvaluator`

**Post-process** (from model output type):
- `torch.Tensor` -> identity
- Has `.logits` attribute -> `lambda x: x.logits` (HuggingFace models)
- Tuple -> `lambda x: x[0]`

### Default Metrics

All metrics that compare against golden outputs are always included when golden
outputs are available.  Target-dependent metrics are added when targets are captured.

| Task Type | Metrics |
|-----------|---------|
| Any (with golden) | PSNR, CosineSimilarity, MSE, AbsErr |
| Classification (with targets) | + TopKAccuracy(k=1), TopKAccuracy(k=5) |
| MLM (with targets) | + MaskedTokenAccuracy |

### Metric Design: `higher_is_better` and Worst Direction

Every `Metric` subclass declares `higher_is_better` which controls how the
worst-case input is identified:

| Metric | higher_is_better | Worst = |
|--------|-----------------|---------|
| PSNR | True | argmin (lowest dB = worst quality) |
| CosineSimilarity | True | argmin (lowest similarity = worst) |
| TopKAccuracy | True | argmin (0.0 = incorrect) |
| MaskedTokenAccuracy | True | argmin (lowest token accuracy) |
| MSE | False | argmax (highest error = worst) |
| AbsErr | False | argmax (highest error = worst) |

### PSNR Cap

PSNR is capped at `PSNR.MAX_PSNR = 100.0` dB.  Raw PSNR above 100 dB (e.g.,
128 dB for near-zero error) is not meaningfully different from perfect match and
creates confusing display.  The cap gives a uniform ceiling: perfect match →
100.0, real degradation → actual dB value below 100.0.

### Per-Sample Statistics

When the dataset has more than one sample, each metric emits additional keys in
the digest alongside the primary mean value:

```
psnr          → mean PSNR across all samples (primary display value)
psnr_min      → worst PSNR sample value
psnr_max      → best PSNR sample value
psnr_worst_idx → dataset index of the worst-performing sample
```

The same pattern applies to all metrics: `{name}_min`, `{name}_max`,
`{name}_worst_idx`.

The frontend renders these as three separate tables:

| Table | Block ID | Content | Shown when |
|-------|----------|---------|------------|
| Accuracy | `accuracy_table` | Mean metric values | Always |
| Per-Sample Stats | `accuracy_stats_table` | `{name}_min` / `{name}_max` per metric | >1 sample |
| Worst Input Index | `accuracy_worst_idx_table` | `{name}_worst_idx` per metric (suffix stripped) | >1 sample |

### Cross-Lens Data Sharing via `_worst_indices`

AccuracyLens exposes the worst-case input indices as class-level state so that
future lenses can access them during their own `observe()` call without
re-running inference:

```python
from executorch.backends.qualcomm.debugger.observatory.lenses.accuracy import AccuracyLens

class PerLayerAccuracyLens(Lens):
    @classmethod
    def observe(cls, artifact, context):
        # Use the worst input identified by AccuracyLens for focused analysis
        worst_idx = AccuracyLens._worst_indices.get("psnr")
        if worst_idx is not None:
            # Run per-layer analysis on dataset[worst_idx]
            ...
```

**Contract:**
- `AccuracyLens._worst_indices` is a `Dict[str, int]` mapping metric name to
  dataset index (e.g., `{"psnr": 3, "cosine_sim": 3, "mse": 7}`)
- Updated after every `evaluate()` call, so it reflects the current record
- Cleared in `_clear_state()` (session end / Observatory.clear())
- Only populated when dataset has >1 sample
- AccuracyLens must be registered before any lens that reads `_worst_indices`
  (lenses run in registration order within each `Observatory.collect()` call)

---

## PerLayerAccuracyLens

### Purpose

Computes sparse per-layer metrics between an anchor graph (default:
`"Exported Float"`) and each collected graph, then renders:

1. Lens-specific graph overlay with raw-PSNR-based coloring.
2. One merged per-layer metrics table (worst -> best).

### Sparse Matching Rule

For each graph:
1. Iterate nodes in topological order.
2. Build key per node:
   - `root:<from_node_root>` when available.
   - `id:<node_name>` fallback when root is missing.
3. Store key -> node using overwrite semantics.

Effect:
- Last topological node for a key is selected (sparse map).
- Pairwise correspondence uses key intersection only.
- No group aggregation.

### Data / Sample Selection

Input sample source:
1. `AccuracyLens._captured_dataset` (primary)
2. `PipelineGraphCollectorLens._last_calibration_dataset` (fallback)

Sample index selection:
1. `config["per_layer_accuracy"]["sample_index"]` if provided.
2. `AccuracyLens._worst_indices` using metric priority list.
3. Fallback index `0`.

### Metrics and Visual Layers

Per matched node:
- `PSNR`
- `CosineSimilarity`
- `MSE`
- `AbsErr`

Graph layers emitted in analyze phase:
- `per_layer_accuracy/psnr` (color by raw `psnr`, low PSNR = severe red)

Default lens graph section:
- `default_layers = ["per_layer_accuracy/psnr"]`
- `default_color_by = "per_layer_accuracy/psnr"`

### Frontend Sections

Record view includes:
1. Summary table.
2. Lens-specific graph section.
3. One merged metrics table with metric-specific column coloring:
   - PSNR column (low PSNR is severe)
   - Cosine column (low cosine is severe)
   - MSE column (high MSE is severe)
   - AbsErr column (high AbsErr is severe)
   - text color is auto-contrasted per cell background

### Config

```python
config = {
    "per_layer_accuracy": {
        "anchor_record_name": "Exported Float",
        # optional explicit sample index
        # "sample_index": 0,
        # optional priority when sample_index is omitted
        "worst_metric_priority": ["psnr", "cosine_sim", "mse", "abs_err"],
    }
}
```

### Registration Note

Register `AccuracyLens` before `PerLayerAccuracyLens` so worst-index hints are
available in the same `collect()` call.

---

## Custom Usage

### Providing a Custom Dataset and Evaluator

For scripts with custom datasets not covered by the auto-patches, users can
provide their own evaluator via the Observatory config:

```python
from executorch.backends.qualcomm.debugger.observatory import Observatory
from executorch.backends.qualcomm.debugger.observatory.lenses.accuracy import (
    StandardEvaluator, TopKAccuracy, PSNR, CosineSimilarity
)

# Prepare your dataset and golden outputs
dataset = [...]  # List of input tuples
targets = [...]  # Ground truth labels
golden = [model(*inp) for inp in dataset]  # Reference outputs

evaluator = StandardEvaluator(
    dataset=dataset,
    metrics=[
        TopKAccuracy(targets, k=1),
        PSNR(golden),
        CosineSimilarity(golden),
    ],
    post_process=lambda x: x.logits,  # optional
)

config = {"accuracy": {"evaluator": evaluator}}

with Observatory.enable_context(config=config):
    build_executorch_binary(model, inputs, ...)
```

### Using MLMEvaluator for Language Models

```python
from executorch.backends.qualcomm.debugger.observatory.lenses.accuracy import (
    MLMEvaluator, MaskedTokenAccuracy, PSNR
)

evaluator = MLMEvaluator(
    dataset=inputs,
    metrics=[MaskedTokenAccuracy(targets), PSNR(golden)],
)

config = {"accuracy": {"evaluator": evaluator}}
with Observatory.enable_context(config=config):
    ...
```

### Custom Metrics

Implement the `Metric` protocol:

```python
class MyMetric:
    def name(self) -> str:
        return "my_metric"

    def calculate(self, predictions: List[torch.Tensor]) -> float:
        # Your metric logic here
        return score

evaluator = StandardEvaluator(
    dataset=dataset,
    metrics=[MyMetric(), PSNR(golden)],
)
```

### Disabling Accuracy Evaluation

Via CLI:
```bash
python -m executorch.backends.qualcomm.debugger.observatory.cli --no-accuracy script.py ...
```

Via config:
```python
config = {"accuracy": {"enabled": False}}
```

---

## Design Notes: AccuracyLens Correctness Invariants

This section documents three correctness issues discovered during development
and the design decisions made to fix them.  They are preserved here so future
contributors understand *why* the code is structured the way it is.

### Problem: Float model shows non-perfect PSNR / cosine similarity

The "Exported Float" record is the golden reference — PSNR should be infinite
(reported as 100.0) and cosine similarity should be 1.0.  Three bugs combined
to violate this invariant.

### Bug 1 — Double `ExportedProgram.module()` call (primary cause)

**Root cause:** `ExportedProgram.module()` creates a **new GraphModule** on
every call via `copy.deepcopy(graph)` + `_unlift_exported_program_lifted_states`.
The golden outputs were computed from GraphModule #1 (inside
`_configure_from_float_model`), but the evaluator called `.module()` again on
the same ExportedProgram, producing GraphModule #2.  Although both share the
same parameter tensors, the deep-copied graph and re-created module can
introduce floating-point non-determinism from different execution ordering.

**Fix:** `_configure_from_float_model` caches the extracted GraphModule as
`_float_model`.  In `observe()`, when the record is "Exported Float", the
cached GraphModule is passed to the evaluator instead of the raw
ExportedProgram.  This guarantees golden == prediction for the float model.

**Execution trace (after fix):**

```
_configure_from_float_model(artifact):
  model = artifact.module()          → GraphModule #1 (cached as _float_model)
  golden = model(dataset)            → golden outputs from GraphModule #1

observe("Exported Float"):
  eval_artifact = cls._float_model   → GraphModule #1 (same instance!)
  evaluator.evaluate(GraphModule #1) → predictions == golden → PSNR=inf, cosine=1.0

observe("Quantized Model"):
  evaluator.evaluate(quantized_gm)   → predictions != golden → shows real degradation
```

### Bug 2 — MLMEvaluator ignored `self.post_process`

**Root cause:** `MLMEvaluator.run_inference()` had hardcoded
`out.logits if hasattr(out, "logits") else out` instead of using
`self.post_process`.  Golden outputs were computed with `_auto_detect_post_process`
(which might return `lambda x: x.logits`, identity, or `lambda x: x[0]`), but
the evaluator used a different extraction path.

For GraphModules from ExportedProgram this happened to work (traced graphs
return plain tensors, so both paths fell through to identity).  But for
original `nn.Module` models (e.g., HuggingFace), the output has `.logits` and
the hardcoded path diverged from whatever `_auto_detect_post_process` detected.

**Fix:** MLMEvaluator now uses `self.post_process(raw_out)` identically to
StandardEvaluator.  The `.logits` extraction is handled by
`_auto_detect_post_process` which returns `lambda x: x.logits` when it detects
a `.logits` attribute on the model output.

### Bug 3 — Missing `torch.no_grad()` in evaluator inference

**Root cause:** `_compute_golden_outputs` runs inside `torch.no_grad()`, but
`StandardEvaluator.run_inference()` and `MLMEvaluator.run_inference()` did not.
The autograd context can cause subtle numerical differences and wastes memory
on gradient tracking during evaluation.

**Fix:** Both evaluators now wrap their inference loop in `torch.no_grad()`.

### Expected metric behavior after fixes

| Pipeline Stage | PSNR | Cosine Sim | MSE | Why |
|---------------|------|------------|-----|-----|
| Exported Float | 100.0 (capped) | 1.0 | 0.0 | Same model instance as golden |
| Annotated Model | 100.0 (capped) | ~1.0 | ~0.0 | Observers in eval mode are pass-through |
| Calibrated Model | 100.0 (capped) | ~1.0 | ~0.0 | Same as annotated (post-calibration) |
| Quantized Model | degraded | < 1.0 | > 0.0 | Q/DQ ops introduce quantization error |
| Edge | degraded | < 1.0 | > 0.0 | Additional lowering transforms |

PSNR is capped at 100.0 — values above 100 dB (e.g., 128 dB for near-zero error
from annotated model observers) are clamped to give a uniform display ceiling.
