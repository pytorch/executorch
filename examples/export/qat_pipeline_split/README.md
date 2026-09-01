# QAT Pipeline Split Example

Shows how to combine ExecuTorch's recipe-based (declarative) export with a
complex, multi-step Quantization-Aware Training flow that lives **outside** the
recipe.

## The problem

`ExportRecipe` / `ExportSession` run the full export pipeline — source
transforms, quantize, torch.export, lower, to-executorch — as a single
declarative call.  That works well when PTQ calibration is sufficient.  Real
QAT workflows are different: they involve training loops, checkpointing, job
restarts, and potentially separate machines for training and compilation.
Those workflows cannot fit inside a single `export()` call.

## The solution: split the recipe around QUANTIZE

The recipe's `pipeline_stages` attribute lets you restrict a session to a
specific subset of stages.  This example uses that mechanism to insert an
arbitrary QAT step between the pre-quantize and post-quantize halves of the
recipe:

```
ExportRecipe (pre-quantize slice)
      |
      v
 .pt2  <-- cross-process / cross-machine handoff
      |
      v
Your QAT training loop  (any framework, any duration)
      |
      v
 .pt2  <-- cross-process / cross-machine handoff
      |
      v
ExportRecipe (post-quantize slice)
      |
      v
 model.pte
```

The recipe still does all the lowering and delegation work; you are only
carving out the QUANTIZE stage to replace it with your own flow.

## Two example modes

### `minimal` — let the recipe adapt to work you already did

Stage 1 captures the eager model with a plain `torch.export.export` call, with
no recipe involved at all.  After your QAT flow (stage 2), you hand the
quantized `.pt2` directly to `export()` with the full recipe.  `ExportSession`
detects that the input is already an `ExportedProgram` and automatically skips
`SOURCE_TRANSFORM`, `QUANTIZE` and `TORCH_EXPORT` — the recipe picks up at
`TO_EDGE_TRANSFORM_AND_LOWER` and completes lowering as normal.

Use this when you have no need for pre-quantize recipe passes and want the
simplest possible seam.

### `sliced` — explicitly slice the recipe's pipeline_stages in two

Stage 1 creates an `ExportRecipe`, restricts `pipeline_stages` to
`[SOURCE_TRANSFORM]`, and runs only that pre-quantize half of the pipeline.
After your QAT flow (stage 2), stage 3 picks up the same recipe, sets
`pipeline_stages` to `[TORCH_EXPORT, TO_EDGE_TRANSFORM_AND_LOWER,
TO_EXECUTORCH]`, and completes the pipeline.  Together the two slices cover
every stage of the recipe with no overlap and no stage silently skipped.

Use this when you need pre-quantize recipe passes to run before QAT (for
example, torchao source transforms that the recipe provides).

### Which should I use?

Use **`minimal`** when the recipe uses its default pipeline stages and you have
no need for pre-quantize recipe passes.  The `minimal` mode works because
`ExportSession` auto-skips SOURCE_TRANSFORM, QUANTIZE and TORCH_EXPORT when it
receives an `ExportedProgram` input — this auto-skip only applies to the
default pipeline.  If the recipe defines a custom `pipeline_stages` list, use
**`sliced`** instead.  The
`sliced` mode is also required whenever pre-quantize recipe passes (such as
torchao source transforms) need to run before handing the graph to your
training loop.

## Stage overview

```
Eager nn.Module
      |
      | Stage 1 -- 1_prepare.py
      |   minimal: torch.export.export (no recipe)
      |   sliced:  ExportRecipe with pipeline_stages=[SOURCE_TRANSFORM]
      v
 stage1_*.pt2
      |
      | Stage 2 -- 2_qat.py  (runs in a separate process)
      |   prepare_qat_pt2e -> dummy train -> checkpoint -> restore -> convert_pt2e
      |   (stand-in for any real QAT training loop)
      v
 stage2_*_quantized.pt2
      |
      | Stage 3 -- 3_lower.py
      |   minimal: ExportRecipe, full recipe (auto-skips pre-quantize stages)
      |   sliced:  ExportRecipe with pipeline_stages=[TORCH_EXPORT,
      |                TO_EDGE_TRANSFORM_AND_LOWER, TO_EXECUTORCH]
      v
 model.pte
      |
      | Stage 4 -- 4_run.py
      |   executorch.runtime: load .pte, run forward, assert output shape
      v
 [1, 10] logits
```

## Files

| File | Role |
|------|------|
| `model.py` | `SmallConvNet` definition (stage 1 only) and example-input helpers (all stages) |
| `1_prepare.py` | Stage 1: capture to ATEN, optionally via a recipe slice |
| `2_qat.py` | Stage 2: QAT + checkpoint save/restore demo |
| `3_lower.py` | Stage 3: lower to `.pte` with the export recipe |
| `4_run.py` | Stage 4: run `.pte` through the ExecuTorch runtime |
| `run.sh` | Bash orchestrator: drives all four stages in sequence |

## Running

```bash
# From the executorch root:
PYTHON=/path/to/python bash examples/export/qat_pipeline_split/run.sh \
    --example minimal \
    --workdir /tmp/qat_pipeline_split_minimal

PYTHON=/path/to/python bash examples/export/qat_pipeline_split/run.sh \
    --example sliced \
    --workdir /tmp/qat_pipeline_split_sliced
```

Each stage can also be run independently:

```bash
python 1_prepare.py --example minimal --workdir /tmp/out
python 2_qat.py     --example minimal --workdir /tmp/out
python 3_lower.py   --example minimal --workdir /tmp/out
python 4_run.py                       --workdir /tmp/out
```

## Expected artifacts in `--workdir`

| File | Written by |
|------|-----------|
| `stage1_minimal.pt2` / `stage1_sliced.pt2` | stage 1 |
| `stage2_minimal_quantized.pt2` / `stage2_sliced_quantized.pt2` | stage 2 |
| `qat_ckpt.pt` | stage 2 (checkpoint demo) |
| `model.pte` | stage 3 |

## Notes

- The concrete backend is XNNPACK (`PT2E_INT8_STATIC_PER_TENSOR`).  To adapt
  to a different backend replace `_build_recipe()` in `1_prepare.py` and
  `3_lower.py`, and `_get_quantizer()` in `2_qat.py`.
- Stage 4 requires the ExecuTorch pybindings (`executorch.runtime`).  If they
  are not installed a warning is printed and the stage exits cleanly.
