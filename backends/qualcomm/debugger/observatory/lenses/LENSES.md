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

---

## PipelineGraphCollectorLens

### Purpose

Automatically collects graph snapshots at each stage of the export -> quantize ->
lower pipeline by monkey-patching framework-level functions. All patches are
framework-level (torchao, executorch.exir), so they work for ALL backends.

### Observation Points

| # | Pipeline Stage | Record Name | Patched Function | Source File | Collected Artifact |
|---|---------------|-------------|-----------------|-------------|-------------------|
| 1 | Export | "Exported Float" | `torch.export.export()` | `torch/export/__init__.py` | Output ExportedProgram |
| 2 | Quantizer Prepare | "Annotated Model" | `prepare_pt2e()` | `torchao/.../quantize_pt2e.py` | Output GraphModule with observers |
| 3 | Quantizer Convert (input) | "Calibrated Model" | `convert_pt2e()` | same | Input GraphModule (post-calibration) |
| 4 | Quantizer Convert (output) | "Quantized Model" | `convert_pt2e()` | same | Output GraphModule with Q/DQ ops |
| 5 | Edge Lowering | "Edge" | `to_edge_transform_and_lower()` | `executorch/exir/program/_program.py` | Output EdgeProgramManager |
| 6 | Edge Transform | "Transformed Edge" | same | same | After transform passes |
| 7 | ETRecord Export | "ETRecord Exported/{method}" | `ETRecord.add_exported_program()` | `executorch/devtools/etrecord/` | Exported program |
| 8 | ETRecord Edge | "ETRecord Edge/{method}" | `ETRecord.add_edge_dialect_program()` | same | Edge dialect program |
| 9 | ETRecord Extra | "ETRecord Extra/{module}" | `ETRecord.add_extra_export_modules()` | same | Extra modules |

### Patching Strategy

Each patch follows the same pattern:
1. Save original function in `_originals[key]`
2. Create wrapper that calls `Observatory.collect(name, artifact)` then calls original
3. Replace function in module namespace via `setattr`
4. On session end, restore all originals

The `to_edge_transform_and_lower` patch also forces `generate_etrecord=True` to
ensure ETRecord collection fires (rows 7-9).

### Additional Data Captured

The `torch.export.export` patch also stores the sample input arguments as
`_last_export_inputs` for use by AccuracyLens as a fallback dataset.

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
| Dataset (inputs) | `get_imagenet_dataset` patch | Sample input from `torch.export.export` args | Custom dataset loader, non-Qualcomm backend |
| Targets (labels) | `get_imagenet_dataset` / `get_masked_language_model_dataset` patch | None (skip target-specific metrics) | Custom dataset, non-classification task |
| Float model | "Exported Float" artifact (ExportedProgram) | -- | Always available |
| Golden outputs | Computed from float model + dataset | -- | Always available if dataset exists |
| Post-process | Auto-detected from model output type | Identity function | Detection failure |

**Fallback behavior when dataset patches don't fire:**
- AccuracyLens uses the single sample input captured by PipelineGraphCollectorLens
- Only PSNR and CosineSimilarity metrics are available (no TopK/MaskedTokenAccuracy)
- This still provides useful signal-to-noise comparison between float and quantized models

### Dataset Loader Patches

| Patched Function | Module | Return Format | What's Captured |
|-----------------|--------|---------------|-----------------|
| `get_imagenet_dataset` | `examples.qualcomm.utils` | `(List[Tuple[Tensor]], List[Tensor])` | inputs + class targets |
| `get_masked_language_model_dataset` | `examples.qualcomm.utils` | `(List[Tuple[Tensor, Tensor]], List[Tensor])` | inputs + masked targets |

### Auto-Detection

**Task type** (from target format):
- Targets contain -100 values -> MLM mode -> uses `MaskedTokenAccuracy` + `MLMEvaluator`
- Otherwise -> Classification mode -> uses `TopKAccuracy` + `StandardEvaluator`

**Post-process** (from model output type):
- `torch.Tensor` -> identity
- Has `.logits` attribute -> `lambda x: x.logits` (HuggingFace models)
- Tuple -> `lambda x: x[0]`

### Default Metrics

| Task Type | Metrics |
|-----------|---------|
| Classification (with targets) | TopKAccuracy(k=1), TopKAccuracy(k=5), PSNR, CosineSimilarity |
| MLM (with targets) | MaskedTokenAccuracy, PSNR, CosineSimilarity |
| No targets (fallback) | PSNR, CosineSimilarity |

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
