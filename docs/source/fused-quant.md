(fused-quant)=

# Fused Quant

::::{note}
Fused Quant is experimental. The package, its recipe building blocks, the portable
recipe and the operator contract may change without notice, at least until a second
backend ships on them. `get_fused_quant_portable_recipe` emits an `ExperimentalWarning`
when called, and a backend recipe built on Fused Quant should be marked the same way.
::::

::::{grid} 2
:::{grid-item-card} What you will learn
:class-card: card-prerequisites
* What the `fused_quant` representation is and why ExecuTorch has it
* What sharing the quantization frontend changes for a backend, and the trade-offs
* How to quantize a model for the portable kernels with it
* How to build a backend's quantization recipe on it
:::
:::{grid-item-card} Before you start
:class-card: card-prerequisites
* {doc}`using-executorch-export`
* {doc}`quantization-overview`
* [torchao PT2E quantization](https://docs.pytorch.org/ao/main/pt2e_quantization/pt2e_quant_ptq.html)
:::
::::

`fused_quant` is a backend-neutral quantization frontend for ExecuTorch. It represents a
`dequantize -> ATen operator -> quantize` pattern as one `torch.ops.fused_quant.*`
operator that carries its quantization parameters with it. In the graph that PT2E
quantization produces, a quantized operator is a float operator between quantize and
dequantize nodes; in a Fused Quant graph, the quantized operator is a single node.

On top of that representation, Fused Quant provides a quantizer, a set of graph
optimizations and a decomposition of the fused operators left after a backend's lowering,
which backends can share. A backend adds what is specific to its hardware: how it transforms the graph to
run on it.

## The representation

Take a quantized convolution. In the quantize/dequantize (q/dq) form that PT2E
quantization produces, the quantized operator is implicit: five nodes, and the one doing
the arithmetic is float. As a `fused_quant` operator it is one node, and the quantization
parameters (qparams) travel with the operands they describe.

```{figure} _static/img/fused_quant/representation.svg
:alt: A quantized convolution as dequantize, aten.convolution and quantize nodes, next to the same convolution as one fused_quant.convolution node carrying the qparams of each operand and of the output
:align: center
```

A few properties make this practical:

- **One schema covers every quantization pattern of an operator.** Each tensor operand,
  and the output, has its own qparam block. A block whose scale is `None` means that edge
  stays in float, so fully quantized, partially quantized (for example, a quantized input
  with a float output) and float instances of an operator are the same operator.
- **Granularity is in the qparams.** Per-tensor, per-channel, per-group and blockwise
  quantization are the same operator with differently shaped `scale` and `zero_point`
  tensors.
- **It converts back.** `DecomposeFusedQuant` turns a fused operator back into the q/dq
  form, so a graph can always return to what PT2E quantization produced.
- **It is ahead-of-time only.** Fused operators have no runtime kernels and never reach a
  `.pte` file: a backend's lowering consumes them, or they are decomposed back to q/dq
  plus ATen operators.

The representation is not tied to a list of operators: any ATen operator can have a
fused counterpart that follows the same contract, named after it and taking its
arguments. The operators defined so far are in `exir/dialects/fused_quant/ops.py`, and
adding one is described in the README.

## What sharing the frontend means for a backend

Besides its lowering, a backend that supports quantized models usually maintains:

- a PT2E quantizer that annotates the graph;
- passes that recover the quantized operators from the q/dq graph that quantization
  produces;
- graph rewrites on those operators.

At the time of writing, 11 backends in this repository ship their own quantizer
classes, and at least 8 run passes that rebuild quantized operators from q/dq nodes.

Fused Quant moves the quantizer, the recovery of quantized operators and the graph
rewrites into shared code, and leaves lowering to the backend.

```{figure} _static/img/fused_quant/stacks.svg
:alt: Separate stacks, where each backend's qspecs feed its own quantizer, then PT2E, then its own recovery of quantized operators from the q/dq graph, graph rewrites and lowering; next to a shared frontend, where each backend's qspecs configure one fused_quant quantizer that also fuses, then PT2E, then shared graph rewrites and each backend's lowering, with no recovery step
:align: center
```

For a backend, this means:

- **Quantization choices are configuration.** A backend expresses its choices with the
  torchao `QuantizationConfig` and `QuantizationSpec` objects every PT2E quantizer uses:
  dtype and range, granularity, symmetric or asymmetric, shared or derived qparams, and
  which operators stay in float. It picks a configuration that matches its hardware and
  lists the operators it wants quantized; the annotation and fusion code is shared.
  Dtype, range and granularity are per-operand arguments of the fused operators, so
  int16, sub-8-bit, per-channel, per-group and blockwise schemes use the same operators.
- **It checks one node at a time.** To decide whether it can run a quantized
  convolution, a backend looks at one node and its qparams, rather than locating a
  dequantize, a float convolution and a quantize and confirming they belong together.
- **Graph rewrites are shared.** Rewrites on quantized operators run on the fused graph
  for every backend: absorbing adjacent quantize/dequantize pairs into an operator,
  hoisting activations above data movement, folding scalar multiplies into neighbouring
  qparams, and cleaning up no-op and cascaded permutes, views, slices, clones and
  concatenations. Rewrites that make a choice for the backend, such as NHWC layout or
  splitting a fused QKV projection, are off by default. A fix to a shared rewrite applies
  to every backend that uses it.
- **Quantization parameters stay attached to operators.** In a q/dq graph, qparams live in
  separate nodes that a backend matches back into patterns or copies into node metadata,
  and a rewrite that doesn't account for them can lose them without an error. As operator
  arguments, every pass and the verifier see them, and removing them gives back the float
  model.
- **Lowering is the backend's own graph transformation.** It is not a one-to-one mapping
  from fused operators to backend operators: a backend transforms the graph however it
  needs to, and the result can contain delegated subgraphs, the backend's own operators,
  ATen operators that run on optimized kernels, or any mix. Only a backend that delegates
  uses a colorer and a partitioner.
- **Remaining fused operators are decomposed.** The steps are sequential: the backend's
  lowering passes run first and produce the operators the backend wants, and then
  `DecomposeFusedQuant` removes any fused operators that remain, since fused operators
  have no kernels. It turns them back into quantize/dequantize plus ATen operators,
  decomposed to core ATen where needed, which run on whichever kernels the program links:
  the portable kernels, an optimized kernel library, or kernels the backend provides. A
  model still compiles when the backend doesn't handle an operator.
- **It can be adopted one operator at a time.** The pipeline with no backend is the
  portable recipe, so a backend in development can run its quantizer end to end before
  it lowers anything, then lower operators one at a time. Operators it doesn't lower take
  the q/dq path an existing flow already consumes, and a delegating backend can keep its
  partitioner, operator builders and compiler. The package depends only on PyTorch,
  torchao and ExecuTorch.

The pipeline figure under [How it fits together](#how-it-fits-together) shows which
stages are shared and which a backend provides.

### Trade-offs and status

- **It is experimental.** The operator contract and recipe building blocks may still
  change, and a backend that adopts them now takes on those changes.
- **Decomposition has a cost.** A fused operator left after lowering runs as
  quantize/dequantize plus ATen operators, which can be slower than a backend-specific
  implementation, especially if only the portable kernels are linked.
- **Coverage is not complete.** Operators without a quantizer rule stay in float, grouped
  and blockwise qparams cannot be decomposed yet, and quantization-aware training is not
  supported.
- **Shared code means shared decisions.** Default rewrites apply to every backend; a
  backend that needs different behavior turns a rewrite off or adds its own pass after
  the shared ones.
- **A delegating backend must lower what it claims.** Coloring and lowering must agree; a
  claimed operator that the backend doesn't lower fails the compile rather than falling
  back.
- **Two backends have prototypes, not yet upstream.** Qualcomm LPAI uses the shared
  quantizer with a configuration matching QNN's 8-bit activation, 8-bit weight scheme,
  keeps QNN's partitioner, builders and preprocess, and produces the same LPAI program as
  the existing QNN flow on MobileNetV2, ResNet18 and two RNN-T models. Arm Ethos-U85
  uses it with Arm's constraints as torchao specs, keeps softmax, the norms, mean,
  division and GELU as whole operators, lowers each claimed operator directly to TOSA,
  and produces quantized graphs 5 to 14 times smaller on a set of nine models.

## How it fits together

Fused Quant plugs into `executorch.export` as an `ExportRecipe`. The figure shows each
stage and who provides it.

```{figure} _static/img/fused_quant/pipeline.svg
:alt: The Fused Quant pipeline from a PyTorch model to to_executorch, with each stage marked as backend-specific, Fused Quant, or standard PyTorch and ExecuTorch
:align: center
```

The quantizer is a torchao PT2E quantizer, so annotation, calibration and conversion are
the standard PT2E flow. What Fused Quant adds is the fusion step after `convert_pt2e`,
the optimizations on the fused graph, and the contract between that graph and a backend.

After optimization, the backend's lowering passes run, then any fused operators that
remain are decomposed:

```{figure} _static/img/fused_quant/fallback.svg
:alt: The optimized fused graph goes through the backend's lowering passes, then DecomposeFusedQuant turns any remaining fused operators into ATen and quantize/dequantize operators; the program can mix delegated subgraphs, backend operators, and ATen and quantize/dequantize operators run by whichever kernels the program links
:align: center
```

## Portable quick start

`get_fused_quant_portable_recipe` runs the whole pipeline with no backend: nothing is
lowered, so every fused operator is decomposed and the program runs on the portable and
quantized kernels. It is the simplest way to try a quantizer, and the starting point for
a new backend.

Start with an ExecuTorch source checkout installed as described in
{doc}`using-executorch-building-from-source`. The example needs torchao and the portable
and quantized kernels, including the quantized operator library loaded by
`executorch.kernels.quantized`; use the full installation.

From the ExecuTorch repository root, run:

```bash
python -m executorch.frontends.fused_quant.quick_start
```

This quantizes a small convolutional model with int8 activations and per-channel
convolution weights, saves `fused_quant_portable.pte`, and runs it with the portable
runtime. It prints the largest absolute difference from the float model on one input.
Use `--output path/to/model.pte` to choose the output file.

The example inputs also form the calibration set. For your model, use representative
calibration data and evaluate accuracy on separate inputs.

```{literalinclude} ../../frontends/fused_quant/quick_start.py
:language: python
:start-at: import argparse
```

## Building a backend on Fused Quant

A backend writes its own `ExportRecipe` from the building blocks in
`frontends/fused_quant/recipe.py`:

- `get_fused_quant_quantization_recipe(quantizer)`: the quantize stage. It captures the
  model, decomposes it to core ATen keeping `quantizer.preserved_ops()` whole, runs PT2E
  with calibration and fuses the result.
- `get_fused_quant_edge_compile_config(preserve_ops)`: the `to_edge` configuration. It
  keeps the given operators whole and lets the edge verifier accept the non-core ones
  among them.
- `get_fused_quant_passes(**flags)`: the shared optimization pipeline, with the optional
  rewrites the backend turns on.

On top of those, every backend supplies a quantizer and a lowering:

1. **A quantizer.** Compose a `FusedQuantQuantizer` from the backend's own quantizers
   followed by `get_core_aten_quantizers(...)`. Annotation is first-match-wins, so a
   backend quantizer that overrides a shared one goes first. To quantize a non-core ATen
   operator whole (`aten.linear`, for example), add an `OpQuantizer` for it; its
   `preserved_ops()` makes the recipe keep that operator out of decomposition.
2. **A lowering.** The passes that transform the fused graph for the backend. They can
   produce delegated subgraphs, the backend's own operators, ATen operators for kernels
   the backend provides or links, or any mix. Once all of them have run,
   `ToChannelsFirst`, `DecomposeFusedQuant` and `AssertNoFusedQuantOps` remove any fused
   operators that remain, which have no kernels, by turning them back into
   quantize/dequantize plus ATen operators.

A backend that delegates also supplies:

3. **A colorer.** Subclass `ColorerBase` and implement `can_lower(ep, node)`: check the
   operator, then any limits on the node's qparams, attributes or users. The colorer
   claims every node the delegate will run, and the shared passes leave claimed nodes
   alone.
4. **A partitioner**, which takes the lowered operators into the delegate.

Put together:

```python
from executorch.backends.transforms.fused_quant.decompose_fused_quant import (
    AssertNoFusedQuantOps,
    DecomposeFusedQuant,
)
from executorch.backends.transforms.fused_quant.optimization_passes.to_channels_first import (
    ToChannelsFirst,
)
from executorch.backends.transforms.fused_quant.passes import get_fused_quant_passes
from executorch.exir._warnings import experimental
from executorch.export import ExportRecipe, LoweringRecipe
from executorch.frontends.fused_quant.recipe import (
    get_fused_quant_edge_compile_config,
    get_fused_quant_quantization_recipe,
)


@experimental("The my_backend fused_quant recipe is experimental.")
def get_my_backend_recipe() -> ExportRecipe:
    quantizer = get_my_backend_quantizer()

    def edge_passes(method_name, program):
        return [
            *get_fused_quant_passes(channels_last=True),
            MyColorer("my_backend"),  # only when delegating
            MyLowering(),
            ToChannelsFirst(),
            DecomposeFusedQuant(keep_whole=quantizer.preserved_ops()),
            AssertNoFusedQuantOps(),
        ]

    return ExportRecipe(
        name="my_backend",
        strict=False,
        quantization_recipe=get_fused_quant_quantization_recipe(quantizer),
        lowering_recipe=LoweringRecipe(
            edge_transform_passes=[edge_passes],
            partitioners=[MyPartitioner()],  # only when delegating
            edge_compile_config=get_fused_quant_edge_compile_config(
                quantizer.preserved_ops()
            ),
        ),
    )
```

Give `DecomposeFusedQuant(keep_whole=...)` and `get_fused_quant_edge_compile_config` the
same operators. A fused operator the backend doesn't lower then ends up exactly as its
float counterpart does after `to_edge`: kept whole if the backend handles that operator
itself, and decomposed to core ATen otherwise.

A backend that delegates must lower every node it claims. The intended way to guarantee
that is to lower the claimed nodes to the backend's own operators before `to_backend`,
so the partitioner takes those operators and has nothing to reject. Whether or not a
backend delegates, `AssertNoFusedQuantOps` catches any fused operator left behind before
it reaches `to_executorch`.

## Graph optimizations

`get_fused_quant_passes()` runs a fixed sequence of constant folding, the optimization
group, and requantize formation. The passes in the optimization group feed each other
(a layout change exposes a fusion, which exposes a dead transpose), so the group repeats
until nothing changes, up to five times. It stops before coloring, so its output doesn't
depend on any backend.

By default the group only canonicalizes and simplifies the graph and keeps its layout.
The rewrites that make a choice for the backend are off until it turns them on:

| Flag | What it does |
|---|---|
| `channels_last` | Runs convolutions and pools in NHWC, between permutes. A backend with its own layout pass would have to undo it. |
| `fold_into_linear` | Folds a scalar add or a constant multiply into the linear or convolution before it. |
| `prequantize_embedding` | Turns an embedding with a quantized table into an int8 gather before any backend sees it. |
| `split_linear_at_slices` | Splits a linear whose output is sliced, such as a fused QKV projection, into one linear per slice. |
| `absorb_quant_with_fork`, `absorb_quant_into_inputs` | Absorb quantization across a fork, or into an operator's inputs. Not always more accurate. |
| `symmetric_quant_input_ops` | Operators whose kernels ignore input zero points, so quantization absorption keeps their inputs symmetric. |

## The operator contract

Backends lower fused operators against a contract, experimental like the rest of the
package:

- **Argument layout.** Quantized operands, then one qparam block per quantized operand in
  the same order, then the `out` block, then everything else in ATen order. A block is
  always `scale`, `zero_point`, `dtype`, `quant_min` and `quant_max`, named
  `<operand>_<field>`.
- **Field meaning.** For an input block, `dtype` is the dequantized float dtype; for the
  `out` block, it is the quantized storage dtype. A `None` scale and zero point mean the
  edge is float.
- **Evolution.** Existing schemas are not edited in place. A new capability is a new
  overload, or a new trailing argument with a default that keeps current behavior.
- **Ahead-of-time only.** Fused operators must never be serialized into a `.pte` file.

## Common questions

**Does a shared quantizer take away backend-specific quantization choices?**
No. Each backend chooses its own configuration of torchao specs to match its hardware,
and adds quantizers for the operators it wants quantized. What is shared is the code that
applies the configuration to the graph. Lowering remains the backend's: it lowers the
fused operators it supports, and any that remain are decomposed afterwards.

**Can a backend keep part of its existing flow?**
Yes. Operators it doesn't lower go back to q/dq plus ATen, which is what a q/dq-based
flow consumes, and a delegating backend can keep its partitioner, builders and preprocess
and lower the fused operators it supports into them.

**What if my target needs a quantization scheme the operators don't express?**
Dtype, range and granularity are per-operand arguments, so many schemes map onto the
existing operators. A capability they don't express is added as a new overload or a new
trailing argument with a default that keeps current behavior, so existing operators are
unchanged.

**When is Fused Quant not a good fit?**
When a backend's quantization flow already covers its models and it has no need for the
shared rewrites or the decomposition, moving to an experimental contract may not be
worth it yet. The same applies to schemes the operators can't express today, such as
grouped or blockwise qparams on operators the backend doesn't lower, and to
quantization-aware training.

## Current limitations

- `DecomposeFusedQuant` rejects grouped and blockwise qparams, so a blockwise operator the
  backend doesn't lower fails the compile.
- The shared quantizers keep bias in float; there is no int32 bias option yet.
- There is no quantization-aware training path: the quantize stage quantizes after
  training, capturing the model in eval mode.
- With partitioners in the `LoweringRecipe`, `to_edge_transform_and_lower` keeps whole
  only the operators the partitioners report, not `EdgeCompileConfig.preserve_ops`, so
  such a partitioner must also report `quantizer.preserved_ops()`.

## Learn more

The [Fused Quant README](https://github.com/pytorch/executorch/blob/main/frontends/fused_quant/README.md) <!-- @lint-ignore -->
has the details: every quantizer, the coloring rules, each pass and its flags, how to add
an operator, and how to run the tests. The code lives in:

- `exir/dialects/fused_quant/`: the operator schemas.
- `frontends/fused_quant/`: the quantizer, the colorer base class and the recipe building
  blocks.
- `backends/transforms/fused_quant/`: the graph optimizations and `DecomposeFusedQuant`.
