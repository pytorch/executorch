# Cortex-M through fused_quant (experimental)

A prototype of the Cortex-M backend on the fused_quant stack: the fused_quant
quantizer and passes, then a lowering of the fused ops to the `cortex_m` CMSIS-NN
kernels. There is no partitioner: the `cortex_m` ops are ordinary kernels, so the
lowering rewrites the edge program in place, and whatever CMSIS-NN cannot run
decomposes to quantize / dequantize plus ATen for the portable kernels.

```python
from executorch.backends.cortex_m.fused_quant.recipe import (
    get_cortex_m_fused_quant_recipe,
)
from executorch.export import export

session = export(
    model=model,
    example_inputs=calibration_inputs,
    export_recipe=get_cortex_m_fused_quant_recipe(),  # an M55 by default
)
session.save_to_pte("model.pte")
```

Layout is explicit, so inputs are contiguous: a model whose inputs are NHWC
starts with a permute to NCHW, which the fused_quant passes cancel against the
one they put ahead of the NHWC kernels.

## Pipeline

1. `quantizer.py`: the current Cortex-M qspecs on the fused_quant quantizer. int8
   per-tensor activations, or another qspec through `activation` (int16 runs
   `div` on its int16 kernel); int8 symmetric weights, per channel for
   convolutions and, with `per_channel_linear`, linears; int32 biases at the
   accumulator scale; softmax outputs on the fixed grid of `arm_softmax_s8`;
   `minimum`, `maximum`, `avg_pool2d`, spatial `mean` and `constant_pad_nd`
   share their input's qparams, as their kernels require. As in the current
   flow, an op reads its producer's grid through data movement, an activation
   after a max pool writes the pool's grid, a convolution or pool no kernel
   runs stays float, and an attention without a mask, dropout or GQA decomposes
   into bmm -> softmax -> bmm ahead of annotation; a masked one runs in float.
2. The fused_quant passes, with `channels_last`, so convolutions and pools reach
   the lowering in the logical layout of the `*_nhwc` kernels.
3. `lowering.py`: `FoldActivations` folds clamping activations into the op
   feeding them; `LowerToCortexM` rewrites each fused op CMSIS-NN can run into
   its `cortex_m` op; `ToChannelsFirst` and `DecomposeFusedQuant` fall the fused
   ops left back; `RetargetToCortexM` moves the int8 quantize, dequantize and
   permute nodes to `cortex_m`; `InitializeScratchBuffersPass` sizes the scratch
   buffers for the target. With no partitioner, nothing is claimed: the fused
   ops lowered first are gone by the time the fallback passes run.

`ops.py` adds the `log` family of fused ops, through the dialect's own
`_define_dq_op_q`. They follow the ATen schemas, so an instance the lowering does
not take decomposes like any other fused op.

## Ops

| fused_quant | cortex_m | Lowered when |
| --- | --- | --- |
| `add`, `sub` | `quantized_add` | same shapes, or one operand of one value per channel (the last dim, or dim 1 of a 4-d tensor); the output shift cannot overflow int32 |
| `mul` | `quantized_mul` | as `add`, no folded activation |
| `div` | `quantized_div` | same shapes, int8 or int16, no folded activation |
| `minimum`, `maximum` | `minimum`, `maximum` | inputs and output on one grid |
| unary ops, scalar `add` / `sub` / `mul` / `div`, `requantize` | `quantized_activation` | int8 in and out |
| `_softmax` | `softmax` (+ `transpose` off the last dim) | up to 4-d, or along the last dim |
| `constant_pad_nd` | `pad` | up to 4-d, no cropping; see below for one channel or one pixel |
| `linear` | `quantized_linear`, `quantized_linear_per_channel` | constant 2-d weight |
| `bmm` | `quantized_batch_matmul` (+ `transpose` for an activation rhs) | 3-d, no folded activation |
| `convolution_channels_last` | `quantized_conv2d_nhwc`, `quantized_depthwise_conv2d_nhwc`, `quantized_transpose_conv2d_nhwc` | 1-d or 2-d; transposed: no groups, dilation or output padding |
| `avg_pool2d_channels_last` | `quantized_avg_pool2d_nhwc` (+ `pad` when padding counts) | one grid, no divisor override |
| `mean.dim` over H and W | `quantized_avg_pool2d_nhwc` over the whole image | 4-d, no dtype, no folded activation |
| `max_pool2d_with_indices_channels_last` | `quantized_max_pool2d_nhwc` | one grid, no dilation or ceil mode, indices unused |
| int8 `quantize_per_tensor`, `dequantize_per_tensor`, `permute_copy` | `quantize_per_tensor`, `dequantize_per_tensor`, `transpose` | |

Every operand must be per tensor, int8 or, for `div`, int16; weights may be per
channel.

The unary ops become a 256-entry table built by running the fused op itself over
every int8 input, so the table is exact for any op and scalar argument; a
requantize table ahead of one composes into it. A 1-d convolution runs as a 2-d
one of height one. A depthwise convolution takes the depthwise kernel only for a
batch of one, and on MVE not for a single input channel with more than 8
outputs, which CMSIS-NN repacks as a regular one. An explicit pad of the input
zero point ahead of a 2-d convolution folds into it when the two make SAME
padding, as the current flow's `FuseConvPaddingPass` folds it. The pad kernel
reads a 4-d tensor of one channel or one pixel as NHWC whatever its layout, as
an NHWC tensor one row high is; such a tensor pads as 3-d, its first two dims
merged, so it stays contiguous.

A softmax, mean or average pool writes a grid of its own: the fixed one, or its
input's. Where quant absorption moved its output to another, a requantize table
follows. Elementwise ops whose operands are all permuted from NHWC read the NHWC
tensors, and their readers get the result back in NCHW only where they need it.

`quantized_conv2d`, `quantized_depthwise_conv2d`, `quantized_transpose_conv2d`,
`quantized_avg_pool2d` and `quantized_max_pool2d` run the same CMSIS-NN kernels
as their `_nhwc` twins on a channels_last dim order. The layout here is explicit,
so the recipe emits the twins.

## Activations

`FoldActivations` folds a `relu`, `hardtanh`, `clamp` or `hardsigmoid` into the
fused op it is the only reader of, through views, permutes and a max pool's
values, as that op's output grid and range. An activation reading grid `A` and
writing grid `B` lets its producer quantize straight to `B`: `Q_B` is monotone,
so clamping before or after it is the same, and the kernels clamp to their
`activation_min` / `activation_max`. The rounding to `A` goes away, as it does
in the current flow.

| Producer | Folds |
| --- | --- |
| convolutions, `linear`, `add`, `sub` | any clamp; `hardsigmoid` as scale × 6 and zero point + round(1 / (2 × scale)) |
| unary ops (tables) | any clamp |
| max pool | a clamp on its own grid, which the quantizer gives it |
| `mul`, `div`, `bmm`, spatial `mean` | a clamp to the dtype's range, which they saturate to |

A fold that would cost the producer its kernel does not happen.

## Tests

`test/test_lowering.py` lowers a model per op and compares the program with the
fused one it came from, both on their reference kernels: exact for tables,
comparisons, copies and max pooling, within one output step for the fixed-point
kernels, and within three where an activation folds, since the fused program
rounds once more. A coverage test checks that every registered `cortex_m` op has
a case.

Run on the Corstone-300 FVP, the programs this recipe builds for the Cortex-M op
and model tests give the outputs their lowered programs give on the reference
kernels, bit for bit, wherever the current flow's programs do.
