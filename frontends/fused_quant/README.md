# fused_quant

> **Experimental.** The package, its recipe building blocks, the portable recipe and the
> op contract may change without notice, at least until a second backend ships on them.
> `get_fused_quant_portable_recipe` is marked `@experimental` and emits an
> `ExperimentalWarning` when called; a backend recipe built on fused_quant should be
> marked the same way.

`fused_quant` is a backend-neutral quantization frontend for ExecuTorch. It represents a
`dequantize -> aten op -> quantize` triple as a single `torch.ops.fused_quant.*` operator
that carries its quantization parameters inline. Instead of a quantized graph being a
float graph sprinkled with q/dq pairs that each backend has to pattern-match and fold
back together, the quantized operator is the unit the graph is built from.

Take a quantized convolution. In the q/dq form the quantized op is implicit: five nodes,
and the one doing the arithmetic is float.

```
  x_i8        w_i8       bias
    |           |          |
    v           v          |
dequantize  dequantize     |      <- scale, zero_point per edge
    |           |          |
    +-----------+----------+
                |
                v
        aten.convolution           <- float compute
                |
                v
             quantize              <- scale, zero_point
                |
                v
              y_i8
```

As a `fused_quant` op it is one node, and the qparams ride along with the operands they
describe:

```
  x_i8        w_i8       bias
    |           |          |
    +-----------+----------+
                |
                v
    fused_quant.convolution
        inp    : scale, zero_point, dtype, quant_min, quant_max
        weight : scale, zero_point, dtype, quant_min, quant_max
        bias   : scale, zero_point, dtype, quant_min, quant_max
        out    : scale, zero_point, dtype, quant_min, quant_max
                |
                v
              y_i8
```

This buys three things:

- **A backend matches one node.** It asks "can I run a quantized convolution with these
  qparams?" by looking at one node, rather than by matching a q/dq pattern around a float
  op, a pattern that every graph rewrite risks breaking.
- **Graph rewrites are written once.** Passes that reason about quantized ops (absorbing
  a requantization into its producer, folding a constant multiply into a weight scale,
  moving a permute through an elementwise op) run on the fused graph for every backend.
- **Nothing is lost for an unsupported op.** `DecomposeFusedQuant` turns the second form
  back into the first, so an op no backend takes still runs on the portable kernels.
  `test_op_numerics.py` checks that an op's reference kernel and its decomposition agree
  exactly.

It is not a delegate. Each operator has a reference (eager) kernel that defines its
numerics, but no runtime kernel: a `fused_quant` op never reaches `to_executorch`. The
frontend produces an edge-dialect graph that a backend colors, lowers and partitions;
anything the backend does not claim is turned back into q/dq + ATen by
`DecomposeFusedQuant` and runs on the ordinary portable kernels. `AssertNoFusedQuantOps`
fails the compile if a fused op is left over.

## How it fits together

```
nn.Module
  |  torch.export in eval mode, then run_decompositions to core ATen,
  |  keeping quantizer.preserved_ops() whole
  v
ATen graph
  |  prepare_pt2e (folds batch norms, annotates) -> calibrate -> convert_pt2e
  v
q/dq graph
  |  FusedQuantQuantizer.fuse: each annotated dq -> op -> q becomes one fused_quant op
  v
fused_quant graph
  |  to_edge (same ops kept whole), then get_fused_quant_passes()
  v
fused_quant edge program
  |  backend: colorer -> lowering to its own ops
  |  then ToChannelsFirst -> DecomposeFusedQuant -> AssertNoFusedQuantOps
  |  then the backend's partitioner and to_backend
  v
to_executorch: delegated ops + q/dq and portable kernels for the rest
```

The quantizer is a torchao PT2E quantizer, so annotation, calibration and conversion are
the standard PT2E flow. What `fused_quant` adds is the fuse step after `convert_pt2e`, the
passes on the fused graph, and the contract (claims, decomposition) between that graph
and a backend.

## Directory structure

```
exir/dialects/fused_quant/
├── ops.py                    fused_quant operator schemas, reference and fake kernels,
│                             QuantParamsStruct
├── ops_utils.py              shape helpers shared by the op definitions
└── test/

backends/transforms/fused_quant/
├── graph_utils.py            qparam / constant / ExportedProgram helpers
├── fuse_aten.py              generic dequant -> op -> quant folding
├── pass_base.py              IterativePassGroup
├── passes.py                 the backend-neutral pass pipeline built from those below
├── decompose_fused_quant.py  DecomposeFusedQuant (back to q/dq + ATen) and
│                             AssertNoFusedQuantOps
├── pre_quantize_passes/      rewrites a quantizer runs before annotation
├── optimization_passes/      rewrites applied to the fused_quant edge graph
└── test/                     including test_op_numerics.py

frontends/fused_quant/
├── colorer.py                CLAIM_KEY, ColorerBase (claims ops for a backend),
│                             ReleaseClaims (backward compatibility only)
├── recipe.py                 the building blocks of a fused_quant ExportRecipe, and
│                             the portable recipe built from them
├── quick_start.py            the portable recipe end to end on a small model
├── quantizer/                torchao PT2E quantizer that also fuses
└── test/
```

## Quick start: the portable kernels

`get_fused_quant_portable_recipe` runs the whole pipeline with no backend: every fused op
decomposes back to q/dq + ATen and the program runs on the portable and quantized
kernels. It is the simplest way to try a quantizer, and the baseline a backend's results
can be compared against. It is experimental and warns when called.

[`quick_start.py`](quick_start.py) quantizes a small convolutional network, saves the
program and runs it:

```bash
python -m executorch.frontends.fused_quant.quick_start --output fused_quant_portable.pte
```

Its core is the quantizer and the export call. `config` and `weighted_config` are
ordinary torchao `QuantizationConfig`s, which the script defines:

```python
quantizer = FusedQuantQuantizer(get_core_aten_quantizers(config, weighted_config))
session = export(
    model=model,
    example_inputs=calibration_inputs,  # list of input tuples; also the calibration set
    export_recipe=get_fused_quant_portable_recipe(quantizer),
)
session.save_to_pte(output_path)
(output,) = session.run_method("forward", (example,))
```

The program keeps `quantized_decomposed` quantize and dequantize ops, which run on the
quantized kernels. `recipe.py` imports `executorch.kernels.quantized`, which registers
them for `to_executorch`.

The example inputs are the calibration set. Calibration runs the model, so the recipe
restores the model's own buffers (a KV cache, for instance) afterwards: both the caller's
model and the exported program start from their initial state.

## Building a backend on fused_quant

A backend writes its own `ExportRecipe`. `recipe.py` supplies the pieces that every
fused_quant recipe needs and that are easy to get subtly wrong, so the backend composes
them rather than copying them:

- `get_fused_quant_quantization_recipe(quantizer)` — the quantize stage: decompose to core
  ATen keeping `quantizer.preserved_ops()` whole, calibrate while restoring the model's own
  buffers, and fuse after `convert_pt2e`. Its output is already an exported graph, so the
  recipe sets `strict=False`.
- `get_fused_quant_edge_compile_config(preserve_ops)` — the `to_edge` config: keep the
  ops whole, let the edge verifier accept the non-core ones among them, and skip
  dim-order ops.
- `get_fused_quant_passes(**flags)` — the optimization pipeline on the fused graph, with
  the rewrites the backend turns on; see [Passes](#passes).

On top of those, the backend supplies:

1. **A quantizer.** Compose a `FusedQuantQuantizer` from the backend's own quantizers
   followed by `get_core_aten_quantizers(...)`. Annotation is first-match-wins, so a
   backend quantizer that overrides a shared one (a different config for some module, a
   `NoopQuantizer` that keeps an op in float) goes first. To keep a non-core ATen op whole
   (`aten.linear`, say), add an `OpQuantizer` for it: its `preserved_ops()` makes the
   recipe hold the op back from decomposition. See [Quantizer](#quantizer).
2. **A colorer.** Subclass `ColorerBase` and implement `can_lower(ep, node)`: check the op,
   then any limits on the node's qparams, attributes or users. The colorer claims every
   node the backend can lower. See [Coloring](#coloring).
3. **A lowering.** Rewrite the claimed nodes to the backend's own ops. Then
   `ToChannelsFirst`, `DecomposeFusedQuant` and `AssertNoFusedQuantOps` turn whatever the
   backend did not claim back into q/dq + ATen, and the backend's partitioner takes its
   own ops, with nothing fused left for it to reject. Give `DecomposeFusedQuant` the
   same `keep_whole` ops as `get_fused_quant_edge_compile_config`, so that an unclaimed
   op ends up as its float counterpart does.

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
            MyColorer("my_backend"),
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
            partitioners=[MyPartitioner()],
            edge_compile_config=get_fused_quant_edge_compile_config(
                quantizer.preserved_ops()
            ),
        ),
    )
```

A backend that keeps an op whole through quantization but wants `to_edge` to decompose
it passes fewer ops to both than the quantizer preserves.

## Operator representation

Every `fused_quant` op takes its tensor operands first, then one qparam block
(`scale`, `zero_point`, `dtype`, `quant_min`, `quant_max`) per quantizable edge, including
the output. A qparam block whose `scale` is `None` means that edge is left in float, so a
single schema covers fully quantized, partially quantized, and float instances of an op,
rather than needing a variant for each.

Granularity is carried by the shape of the `scale` tensor rather than by separate op
variants or an `axis` argument. The qparams are always shaped to broadcast against the
tensor they describe:

| Granularity | `scale` shape for a `[C_out, C_in, kH, kW]` weight | Rule |
|---|---|---|
| per-tensor | `[]` or `[1, 1, 1, 1]` | `scale.numel() == 1` |
| per-channel | `[C_out, 1, 1, 1]` | exactly one non-unary dim, block size 1 along it |
| per-group / blockwise | e.g. `[C_out, C_in // g, 1, 1]` | `tensor.shape[i] // scale.shape[i]` per dim |

Test granularity with `QuantParamsStruct.is_per_tensor()`, `is_per_channel(tensor_shape)`
and `is_per_group(tensor_shape)`, not the shape. The last two take the quantized tensor's
shape, so a group scale such as `[1, K // g]` over `[1, K]` is not mistaken for
per-channel. `channel_axis()` returns the per-channel axis.

Each op's fake kernel reports the dtype and shape its eager kernel returns, including for
a partially quantized instance (quantized input, float output).

The ops are listed in `ops.py`. Today they cover `convolution` (plus `conv1d`, `conv2d`,
`conv3d`, which `ToConvolution` canonicalizes to it), `linear`, `bmm`, `embedding`,
`add`/`sub`/`mul` (tensor and `.Scalar`), the activations (`relu`, `gelu`, `sigmoid`,
`tanh`, `hard_tanh`, `hardswish`, `hardsigmoid`, `silu`), `max_pool2d_with_indices`,
`avg_pool2d`, `native_layer_norm`, `rms_norm`, and `requantize`. The
`*_channels_last` variants of convolution and the pools exist only between
`ToChannelsLast` and `ToChannelsFirst`.

## Numerics

The reference (eager) kernel of each op defines its numerics, and every one has the same
shape:

1. Dequantize each quantized operand with torchao's `dequantize_affine` to the float
   dtype its qparam block names: `(q - zero_point) * scale`.
2. Run the ATen op in float.
3. Quantize the output with `quantize_affine`:
   `clamp(round(x * (1 / scale)) + zero_point, quant_min, quant_max)`, where `round`
   rounds half to even.

The details a backend has to match, or knowingly deviate from:

- **Qparam dtypes.** Fusion stores a per-tensor scale as a 0-d `float32` constant and its
  zero point as a 0-d `int64` constant; PT2E's per-tensor q/dq ops carry the scale as a
  Python float. Per-channel and affine qparams pass through as `convert_pt2e` produced
  them, reshaped to full rank.
- **Bias.** A quantized bias is dequantized with its own qparams, like any operand. The
  shared quantizers leave bias in float (a null qparam block), so the reference adds it
  in float. An integer backend that adds the bias to its int32 accumulator, at scale
  `inp_scale * weight_scale`, quantizes it itself; the reference does not model that
  rounding.
- **Decomposition is exact.** `DecomposeFusedQuant` emits q/dq + ATen that computes the
  same values as the reference kernel, bit for bit; `test_op_numerics.py` checks this for
  every op.
- **The default passes round less, not more.** Output-side quant absorption folds a
  `dequantize -> quantize` after an op into the op's output qparams, rounding once where
  the q/dq graph rounds twice. The fused program is therefore not bit-exact with the
  graph `convert_pt2e` produced, but never less accurate. The rewrites that do drop or
  move a rounding in a way that can lose accuracy (`fold_relu_into_producer`,
  `fold_into_linear`, and quant absorption's `absorb_quant_with_fork` and
  `absorb_quant_into_inputs` modes) are off by default.
- **No tolerance is defined.** A backend's integer kernels generally differ from the
  reference in accumulation and requantization rounding. How close is close enough is up
  to each backend's tests.

## Operator contract

Backends lower these ops against the following contract, which is experimental like the
rest of the package:

- **Argument layout.** Quantized operands, then one qparam block per quantized operand
  in the same order, then the `out` block, then everything else in ATen order. A block
  is always the five fields above, named `<operand>_<field>`. Tensor operands without a
  block come after `out`: `embedding`'s `indices`, and the `weight` and `bias` of
  `native_layer_norm` and `rms_norm`.
- **Field meaning.** For an input block, `dtype` is the dequantized float dtype; for the
  `out` block, it is the quantized storage dtype. A `None` scale/zero point means the edge
  is float.
- **Numerics.** The reference kernel, as described under [Numerics](#numerics).
- **Evolving an op.** Existing schemas are not edited in place. A new capability is a new
  overload, or a new trailing argument with a default that preserves current behavior.
- **Ahead-of-time only.** `fused_quant` ops have no runtime kernels and must never be
  serialized into a `.pte`. `AssertNoFusedQuantOps` enforces this.

## Quantizer

`quantizer/quantizer.py` defines a torchao `Quantizer` that both annotates and fuses:

- `QuantizerBase` — the interface: annotate nodes, fuse the annotated ones, and report
  `preserved_ops()`.
- `OpQuantizer` — annotates one ATen op type and folds matched instances into a
  `fused_quant` op. Takes a `QuantizationConfig` plus the schema names of its quantized
  activations, weights, and passthrough operands (such as `bias`), and an optional
  `node_filter`. Most quantizers are instances of this rather than subclasses; subclass
  it only when an op needs custom annotation or fusion. Its `config_for`,
  `activation_names_for` and `fused_op_for` hooks vary those per node.
- `BinaryQuantizer` — `add`, `sub` and `mul`, whose second operand is a tensor or a
  Python scalar. `x + 1e-5` traces to `aten.add.Tensor` with a scalar `other`, so a node
  with a scalar quantizes only `input` and fuses into the `.Scalar` fused op.
- `ConvolutionQuantizer` — puts a transposed convolution's per-channel weight scales on
  axis 1, where its output channels are; a grouped transposed convolution gets
  per-tensor weights instead.
- `MaxPoolQuantizer` — `max_pool2d_with_indices`, whose pooled values share the input's
  observer.
- `NoopQuantizer` — claims nodes with an empty annotation so a later quantizer skips them,
  leaving the op in float.
- `FusedQuantQuantizer` — composes the above. Annotation is first-match-wins, so a narrower
  quantizer must be listed before the general one it overrides. Before annotation, its
  `transform_for_annotation` runs each child's own, in order. Its `fuse` runs after
  `convert_pt2e` and first repairs two gaps in torchao's output that fusion trips on.
- `get_core_aten_quantizers(config, weighted_config)` — op quantizers every backend can
  share: the binary ops, `bmm`, the activations, the pools, `convolution` and
  `native_layer_norm`. Convolution uses `weighted_config`; everything else uses
  `config`. Every op they quantize is core ATen, so each one sees its op after
  decomposition and anything left unclaimed decomposes to portable kernels. It is not
  every core ATen op. A backend adds its own quantizers for the non-core ops it wants to
  keep whole.
- `module_fqn_filter(pattern)` — a `node_filter` that matches nodes by module name, for
  quantizing part of a model differently.

`preserved_ops()` reports the ops the quantizer matches. The recipe building blocks keep
them intact through `run_decompositions` (`QuantizationRecipe.preserve_ops`) and `to_edge`
(`EdgeCompileConfig.preserve_ops`). Core ATen ops survive decomposition anyway, so keeping
ops intact only matters for the non-core ops a backend adds. A backend that lowers an op
whole without quantizing it, or rewrites it before annotation, reports it the same way,
from a child quantizer that annotates nothing.

`prepare_pt2e` folds a batch norm into the convolution before it, including the
`aten.convolution` that `conv1d`, `conv2d` and `conv3d` decompose to, so a
convolution followed by a batch norm is quantized as one convolution. Folding into
`aten.convolution` needs a torchao that includes pytorch/ao#5102; with an older one,
the batch norm stays in float between the fused convolution and its consumer, unless the
backend's quantizer runs `FoldBatchNorm`.

## Coloring

`ColorerBase` is the step between optimization and lowering. A backend implements
`can_lower(ep, node)`, which says whether it can lower a node: its op, then any limits on
the node's qparams, attributes or users. The base class walks the graph and claims every
node the backend can lower by setting `node.meta[CLAIM_KEY]` to the backend's name. The
claim has its own key rather than `delegation_tag`, which partitioners write per partition
and `to_backend` consumes. Colorers compose by priority: the first to claim a node keeps
it, and a later colorer skips claimed nodes.

The claim is a statement of ownership, and the generic passes honour it: `ToChannelsLast`
and `ToChannelsFirst` will not relayout a claimed node, and `DecomposeFusedQuant` will not
decompose one. So the pipeline is colorer -> `ToChannelsFirst` -> `DecomposeFusedQuant`:
the backend keeps what it claimed, and everything it did not claim falls back to q/dq
plus ATen. A backend describes what it supports, not what it does not.

A backend that claims nothing is the degenerate case: every fused op decomposes and the
program runs entirely on the portable ATen kernels. That is the portable recipe.

A backend must lower every node it claims: coloring and partitioning are meant to match.
The intended way to guarantee that is to lower the claimed nodes to the backend's own ops
before `to_backend`, so its partitioner takes those ops and has nothing to reject. Run
`AssertNoFusedQuantOps` before `to_executorch` to catch any fused op left behind.

`ReleaseClaims` is not part of that design. It exists only for backward compatibility
and to ease the transition of backends whose partitioner can still reject a claimed node,
which `to_backend` then leaves fused and still claimed. For those, once every backend has
run `to_backend`, run `ReleaseClaims`, then `ToChannelsFirst` and `DecomposeFusedQuant`
again, to fall the rejected nodes back to the portable kernels. New backends should not
rely on it.

## Passes

### Before annotation

`pre_quantize_passes/` holds rewrites a backend's quantizer runs from its
`transform_for_annotation`, which `prepare_pt2e` calls before annotation.
`FoldBatchNorm` folds a batch norm into the convolution before it, the fold
`prepare_pt2e` already does before calling `transform_for_annotation` with a torchao
that includes pytorch/ao#5102; with one, it finds nothing left to fold.

### On the fused graph

`get_fused_quant_passes()` in `passes.py` is the pipeline a fused_quant recipe runs on
the fused edge graph: `ConstantFold`, then the optimization group, then `ConstantFold` and
`ReplaceDequantQuantWithRequantize`. It stops before coloring, so its output is
hardware-independent.

The optimization group (`get_optimization_passes()`) is an `IterativePassGroup`. The
passes feed each other (a layout change exposes a fusion, which exposes a dead
transpose), so the group re-runs until no pass reports a change, up to five iterations;
a graph still changing on the last one is left as-is, with a warning.

By default the group only canonicalizes and simplifies the graph and keeps its layout:
it canonicalizes `conv1d`/`conv2d`/`conv3d` to `convolution` and `sub` of a constant to
`add`, absorbs adjacent q/dq into the fused ops' output qparams, hoists activations above
data movement, folds positive scalar multiplies into neighboring qparams, and removes or
merges no-op and cascaded permutes, views, slices, clones and cats. The rewrites that
make a choice for the backend are off until it turns them on:

| Flag | Pass | What it does |
|---|---|---|
| `channels_last` | `ToChannelsLast` | Runs convolutions and pools in NHWC, between permutes. A backend with its own layout pass would have to undo it. |
| `fold_relu_into_producer` | `FoldReluIntoProducer` | Has the convolution or linear before a relu quantize straight to the relu's output, dropping the rounding between the two; the relu is removed when it no longer clamps anything. |
| `fold_into_linear` | `FuseAddIntoLinear`, `FuseMulIntoLinear` | Folds a scalar add or a constant multiply into the linear or convolution before it, dropping the rounding between the two. |
| `prequantize_embedding` | `PrequantizeEmbedding` | Turns an embedding with a quantized table into an int8 gather before any backend sees it. |
| `split_linear_at_slices` | `SplitLinearAtSlices` | Splits a linear whose output is sliced, such as a fused QKV projection, into one linear per slice. |
| `absorb_quant_with_fork`, `absorb_quant_into_inputs` | `QuantAbsorptionPass` | Absorb quantization across a fork, or into an op's inputs. Not always more accurate; see `QuantAbsorptionPass`. |
| `symmetric_quant_input_ops` | `QuantAbsorptionPass` | Ops whose kernels ignore input zero points: absorption keeps their inputs symmetric. |
| `extra_quantizable_ops` | `AdvanceQuantizeOpAboveDefChainPass` | Further ops a quantize may move above, each with the indices of the inputs it moves to. |

`FUSED_QUANT_ELEMENTWISE` lists the fused elementwise ops that
`RemovePermutesAroundElementwiseOps` may move permutes through. The view and slice passes
have their own lists in `get_optimization_passes`: `FuseViewCopyTransform`'s
`extra_fusible_ops` and `PropagateSlice`'s `additional_binary_targets`.

### Back to q/dq

`decompose_fused_quant.py` is the inverse of fusion: `DecomposeFusedQuant` turns a fused
op back into dequantize + ATen + quantize. It works from the op rather than a fixed list:
the ATen op is the one with the fused op's name (`hard_tanh` finds `hardtanh`), the
overload is the one whose trailing arguments match, and the qparam blocks come from the
fused op's schema. It handles multi-output ops through `getitem`, and validates every
node before rewriting any, so an unsupported one fails the pass without leaving a partly
decomposed graph. Constant per-channel qparams become 1-D constants, and qparam constants
left without users are removed. Because the ATen op is found by name, it only knows the
ATen layouts: run `ToChannelsFirst` first to bring `*_channels_last` ops back.

An unclaimed op ends up as its float counterpart does after `to_edge`. When its ATen op
is not core ATen (`aten.linear`, say), `DecomposeFusedQuant` keeps it whole if it is in
`keep_whole`, which `get_fused_quant_edge_compile_config` also passes to the edge verifier
as `EdgeCompileConfig._core_aten_ops_exception_list`. Otherwise it decomposes the op to
core ATen with the table `to_edge` uses. The portable recipe keeps nothing whole, so its
program only needs the portable and quantized kernels.

## Adding an operator

1. **Define the op** in `exir/dialects/fused_quant/ops.py` with
   `@torch.library.custom_op("fused_quant::<name>", mutates_args=())`. Name it after the
   ATen op (underscores may be added, as in `hard_tanh`), and lay out its arguments as the
   [contract](#operator-contract) says, ending with the ATen op's own arguments under
   their ATen names: `DecomposeFusedQuant` finds the ATen overload by name and by those
   trailing arguments. The eager kernel dequantizes each operand, runs the ATen op and
   quantizes the result (`_maybe_dequantize`, `_maybe_quantize`); register a fake kernel
   that returns the same dtype and shape, including when only some edges are quantized.
2. **Quantize it** with an `OpQuantizer(aten_op, fused_op, config, activation_names=...,
   weight_names=..., other_names=...)`, or a subclass if it needs custom annotation or
   fusion. Add it to `get_core_aten_quantizers` only if the ATen op is core and every
   backend should quantize it; otherwise it belongs in the backend's quantizer list.
3. **Let the passes see through it.** Add an elementwise op to `FUSED_QUANT_ELEMENTWISE`
   so permutes can move across it, and to the view or slice lists in
   `get_optimization_passes` if views or slices should too.
4. **Test it.** Add a case to `test_op_numerics.py`, which checks that the fake kernel
   matches the eager one and that the decomposition matches the reference exactly, and
   a quantizer test that fuses it from an exported model.

## Tests

From the ExecuTorch root:

```bash
pytest exir/dialects/fused_quant/test backends/transforms/fused_quant/test frontends/fused_quant/test
```

`backends/transforms/fused_quant/test/test_op_numerics.py` checks every op against its
decomposition, `frontends/fused_quant/test/test_recipe.py` exports through the portable
recipe and runs its program, and through a backend-style recipe built from the pieces,
`frontends/fused_quant/test/test_quick_start.py` runs the quick start, and each pass has a
test of its own.

## Known gaps

- `DecomposeFusedQuant` rejects grouped and blockwise qparams rather than emitting
  `dequantize_affine`, so an unclaimed blockwise op fails the compile.
- The shared quantizers pass bias through in float; there is no int32 bias option yet.
- There is no QAT path: `get_fused_quant_quantization_recipe` quantizes post-training,
  capturing the model in eval mode.
- `test_op_numerics.py` uses per-tensor qparams with zero point 0, so it does not cover
  per-channel qparams or nonzero zero points.
- There is no numerics tolerance a backend is held to; see [Numerics](#numerics).
- Ops register on import, and `passes.py` looks up `exir_ops.edge.fused_quant.*` at
  module scope.
- Schemas are written by hand rather than generated from the ATen ones.
- There is no generic `Partitioner` for claimed nodes: a backend's own partitioner has
  to accept the fused ops it claimed.
- With partitioners in the `LoweringRecipe`, `to_edge_transform_and_lower` keeps whole
  only the ops the partitioners' `ops_to_not_decompose` return, not
  `EdgeCompileConfig.preserve_ops`. Until it honors both, such a partitioner must also
  report `quantizer.preserved_ops()`.
