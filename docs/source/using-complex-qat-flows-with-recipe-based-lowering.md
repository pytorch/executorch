# Using Complex QAT Flows with Recipe-Based Lowering

How to split the declarative export recipe around the quantization boundary so
that an arbitrary Quantization-Aware Training flow can live outside of — and
between the two halves of — the recipe pipeline.

## Motivation

`ExportRecipe` and `ExportSession` run the full export pipeline — source
transforms, quantize, torch.export, lower, to-executorch — as a single
declarative call.  That works well for simple calibration flows.

Complex QAT is different.  A real training run involves iterative training
loops, periodic checkpointing, job restarts, accuracy evaluation on validation
datasets, rollback when a checkpoint regresses, and potentially separate
machines for training and for compilation.  None of that can fit inside a single
`export()` call.

## How It Works

The recipe's `pipeline_stages` attribute lets you restrict an `ExportSession`
to a specific subset of stages.  Use it to split the pipeline around the
`QUANTIZE` stage: run everything up to (but not including) quantization in one
recipe invocation, hand the result off to your QAT flow as a `.pt2` file, and
then continue with the post-quantize stages in a second recipe invocation.

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

The recipe still owns all the lowering and delegation work; you are only
carving out the `QUANTIZE` stage to replace it with your own flow.

## The Method

### Pre-quantize slice

Run only the pre-quantize stages through `ExportSession`, then export the
resulting module to an ATEN-dialect `ExportedProgram` and save it as a `.pt2`
handoff file:

```python
from executorch.export import export as et_export
from executorch.export.types import StageType

recipe.pipeline_stages = [StageType.SOURCE_TRANSFORM]
sess = et_export(model, example_inputs=[inputs], export_recipe=recipe)

transformed = sess.get_stage_artifacts()[StageType.SOURCE_TRANSFORM].data["forward"]

ep = torch.export.export(transformed, inputs, strict=True)
torch.export.save(ep, "pre_qat.pt2")
```

> **Simplification:** if you do not need any pre-quantize recipe passes (such
> as source transforms provided by the recipe), you can skip this half entirely.
> Capture the eager model directly with `torch.export.export` and save the
> result as the handoff `.pt2`. No recipe is involved in that case.

### Your QAT flow

Load the handoff, prepare it for QAT, and train:

```python
# Load the pre-quantize graph.
captured_gm = torch.export.load("pre_qat.pt2").module()

# Prepare for QAT once.
from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e
from torchao.quantization.pt2e import move_exported_model_to_train

prepared = prepare_qat_pt2e(captured_gm, quantizer)
move_exported_model_to_train(prepared)

# Train -- this is an open-ended process.
# It can be paused and resumed from checkpoints.
# It can run for hours, days, or weeks.
# It can be distributed across separate machines or jobs.
# Evaluate accuracy on a validation dataset periodically and
# roll back to an earlier checkpoint whenever accuracy regresses.
# Save checkpoints as you go:
#   torch.save(prepared.state_dict(), "qat_ckpt.pt")
# Resume from a checkpoint:
#   prepared.load_state_dict(torch.load("qat_ckpt.pt"))

# Only once you are satisfied with the result:
from torchao.quantization.pt2e import move_exported_model_to_eval
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e

move_exported_model_to_eval(prepared)
quantized = convert_pt2e(prepared)

ep = torch.export.export(quantized, inputs, strict=True)
torch.export.save(ep, "post_qat.pt2")
```

### Post-quantize slice

Continue the recipe pipeline from `TORCH_EXPORT` onward:

```python
recipe.pipeline_stages = [
    StageType.TORCH_EXPORT,
    StageType.TO_EDGE_TRANSFORM_AND_LOWER,
    StageType.TO_EXECUTORCH,
]
quantized_gm = torch.export.load("post_qat.pt2").module()
sess = et_export(quantized_gm, example_inputs=[inputs], export_recipe=recipe)
sess.save_to_pte("model")  # writes model.pte
```

> **Simplification:** if the recipe uses its default pipeline stages, you do
> not need to set `pipeline_stages` at all on the lowering side.  Pass the
> `.pt2` file path directly to `export()` and `ExportSession` will
> automatically skip `SOURCE_TRANSFORM`, `QUANTIZE`, and `TORCH_EXPORT` when it
> detects an `ExportedProgram` input, picking up at
> `TO_EDGE_TRANSFORM_AND_LOWER`.  This auto-skip only applies to the default
> pipeline; if the recipe defines a custom `pipeline_stages` list, set the
> post-quantize stages explicitly as shown above.

## Example

A runnable end-to-end example demonstrating both the full split and the
simplified path is available at
[`examples/export/qat_pipeline_split/`](https://github.com/pytorch/executorch/tree/main/examples/export/qat_pipeline_split).

See the [example README](https://github.com/pytorch/executorch/blob/main/examples/export/qat_pipeline_split/README.md)
for full setup instructions, expected artifacts, and notes on adapting the
example to a different backend.
