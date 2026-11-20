# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lowerings of the elementwise fused_quant ops, softmax and pad."""

import math
from collections.abc import Callable
from typing import cast, Optional

import executorch.backends.cortex_m.fused_quant.ops  # noqa: F401
import torch
from executorch.backends.cortex_m.fused_quant.utils import (
    Context,
    INT8_MAX,
    INT8_MIN,
    QParams,
    requantize_table,
    Rule,
    Rules,
    shape,
)
from executorch.backends.cortex_m.passes.aten_to_cortex_m_pass import (
    _compute_softmax_params,
)
from executorch.backends.cortex_m.quantizer.quantization_configs import (
    CMSIS_SOFTMAX_SCALE,
    CMSIS_SOFTMAX_ZERO_POINT,
)
from executorch.backends.cortex_m.utils import (
    is_channels_last,
    quantize_multiplier_aot,
    quantize_val,
    SHIFT_INT8,
    to_physical_order,
)
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from torch import fx
from torch._prims_common import make_contiguous_strides_for
from torch.fx.node import Argument, map_arg

_aten = exir_ops.edge.aten
_cortex_m = exir_ops.edge.cortex_m
_fused_quant = exir_ops.edge.fused_quant

_NCHW_TO_NHWC = [0, 2, 3, 1]
_NHWC_TO_NCHW = [0, 3, 1, 2]


def _unary_qparams(ctx: Context, node: fx.Node) -> Optional[tuple[QParams, QParams]]:
    inp, out = ctx.qparams(node, "inp"), ctx.qparams(node, "out")
    return None if inp is None or out is None else (inp, out)


def _binary_qparams(
    ctx: Context,
    node: fx.Node,
    dtypes: tuple[torch.dtype, ...] = (torch.int8,),
) -> Optional[tuple[QParams, QParams, QParams]]:
    inp, other = ctx.qparams(node, "inp", dtypes), ctx.qparams(node, "other", dtypes)
    out = ctx.qparams(node, "out", dtypes)
    if inp is None or other is None or out is None:
        return None
    return inp, other, out


def _same_shapes(node: fx.Node) -> bool:
    return shape(node.args[0]) == shape(node.args[1]) == node.meta["val"].shape


def _operand_layout(node: fx.Node) -> Optional[str]:
    """How the add and mul kernels take a binary node's operands, if they can.

    "same" for equal shapes. Otherwise the kernels broadcast an operand that
    holds one value per channel of the other, the channels being innermost in
    memory: "innermost" when they are the last dim, "nchw" when they are dim 1
    of a 4-d tensor, which then runs in NHWC.
    """
    a, b = shape(node.args[0]), shape(node.args[1])
    out = node.meta["val"].shape
    if a == b == out:
        return "same"
    if out not in (a, b):
        return None
    small = b if a == out else a
    if out[-1] > 1 and math.prod(small) == small[-1] == out[-1]:
        return "innermost"
    if (
        len(out) == 4
        and out[1] > 1
        and len(small) >= 3
        and math.prod(small) == small[-3] == out[1]
    ):
        return "nchw"
    return None


def _is_permute(node: object, dims: list[int]) -> bool:
    return (
        isinstance(node, fx.Node)
        and node.target == _aten.permute_copy.default
        and list(cast(list[int], node.args[1])) == dims
    )


def _erase_if_unused(node: fx.Node) -> None:
    if not node.users:
        node.graph.erase_node(node)


def _nhwc_sources(operands: tuple[fx.Node, ...]) -> Optional[list[fx.Node]]:
    """The NHWC tensors behind operands, if each is a permute to NCHW that
    only the elementwise op reading it reads."""
    if not all(
        _is_permute(operand, _NHWC_TO_NCHW) and len(operand.users) == 1
        for operand in operands
    ):
        return None
    return [cast(fx.Node, operand.args[0]) for operand in operands]


def _hand_over_nhwc(ctx: Context, node: fx.Node, result: fx.Node) -> None:
    """Replace node, which wrote NCHW, with result, its NHWC image: readers
    permuting it to NHWC read result itself, the rest a permute back."""
    nchw = None
    for user in list(node.users):
        if _is_permute(user, _NCHW_TO_NHWC):
            user.replace_all_uses_with(result)
            node.graph.erase_node(user)
            continue
        if nchw is None:
            nchw = ctx.call(node, _aten.permute_copy.default, (result, _NHWC_TO_NCHW))
        user.replace_input_with(node, nchw)
    node.graph.erase_node(node)


def _lower_binary(
    ctx: Context,
    node: fx.Node,
    target: EdgeOpOverload,
    args: Callable[[fx.Node, fx.Node], tuple[Argument, ...]],
) -> None:
    """Replace node with ``target(*args(lhs, rhs))``, its operands laid out as
    the add and mul kernels take them.

    A broadcast runs over ``[rows, channels, 1, 1]`` views, the shape whose
    second dim the kernels broadcast along: they add the per-channel operand to
    each run of ``channels`` contiguous values. An NCHW broadcast runs on the
    NHWC tensor behind the permute that made it NCHW, or a permute of it, and
    its result goes back to NCHW unless every reader permutes it to NHWC.
    Without a broadcast, operands both permuted from NHWC are read as such.
    """
    layout = _operand_layout(node)
    assert layout is not None
    lhs, rhs = cast(tuple[fx.Node, fx.Node], node.args[:2])
    if layout == "same":
        sources = _nhwc_sources((lhs, rhs))
        if sources is None:
            ctx.replace(node, target, args(lhs, rhs))
            return
        _hand_over_nhwc(ctx, node, ctx.call(node, target, args(*sources)))
        for operand in dict.fromkeys((lhs, rhs)):
            _erase_if_unused(operand)
        return

    out_shape = list(node.meta["val"].shape)
    big_first = list(shape(lhs)) == out_shape
    big, small = (lhs, rhs) if big_first else (rhs, lhs)
    bypassed = None
    if layout == "nchw":
        channels = out_shape[1]
        if _is_permute(big, _NHWC_TO_NCHW):
            bypassed, big = big, cast(fx.Node, big.args[0])
        else:
            big = ctx.call(node, _aten.permute_copy.default, (big, _NCHW_TO_NHWC))
    else:
        channels = out_shape[-1]
    rows = math.prod(out_shape) // channels
    big = ctx.call(node, _aten.view_copy.default, (big, [rows, channels, 1, 1]))
    small = ctx.call(node, _aten.view_copy.default, (small, [1, channels, 1, 1]))
    result = ctx.call(node, target, args(big, small) if big_first else args(small, big))

    if layout == "innermost":
        result = ctx.call(node, _aten.view_copy.default, (result, out_shape))
        node.replace_all_uses_with(result)
        node.graph.erase_node(node)
    else:
        nhwc_shape = [out_shape[0], *out_shape[2:], channels]
        result = ctx.call(node, _aten.view_copy.default, (result, nhwc_shape))
        _hand_over_nhwc(ctx, node, result)
    if bypassed is not None:
        _erase_if_unused(bypassed)


# -----------------------------------------------------------------------------
# Unary ops: a 256-entry lookup table
# -----------------------------------------------------------------------------


def _can_lower_lut(ctx: Context, node: fx.Node) -> bool:
    inp = node.args[0]
    return _unary_qparams(ctx, node) is not None and all(
        arg is inp or ctx.constant(arg) is not None for arg in node.all_input_nodes
    )


def _lower_lut(ctx: Context, node: fx.Node) -> None:
    """Tabulate the fused op itself over every int8 input.

    The table is then the op's exact int8 image, whichever op it is and
    whatever its scalar arguments. A table read only by this one, such as a
    requantize ahead of an activation, composes into it. An input permuted
    from NHWC is read as such.
    """
    inp = cast(fx.Node, node.args[0])
    values = torch.arange(INT8_MIN, INT8_MAX + 1).to(torch.int8)

    def concrete(arg: fx.Node) -> torch.Tensor:
        if arg is inp:
            return values
        constant = ctx.constant(arg)
        assert constant is not None
        return constant

    target = cast(EdgeOpOverload, node.target)
    lut = target(*map_arg(node.args, concrete), **map_arg(node.kwargs, concrete))
    nhwc = _nhwc_sources((inp,))
    source = inp if nhwc is None else nhwc[0]
    composed = (
        source.target == _cortex_m.quantized_activation.default
        and len(source.users) == 1
    )
    table_input = source
    if composed:
        inner = ctx.constant(source.args[1])
        assert inner is not None
        lut = lut[inner.to(torch.int64) - INT8_MIN]
        table_input = cast(fx.Node, source.args[0])
    result = ctx.call(
        node,
        _cortex_m.quantized_activation.default,
        (table_input, ctx.add_constant(node, "lut", lut)),
    )
    if nhwc is None:
        node.replace_all_uses_with(result)
        node.graph.erase_node(node)
    else:
        _hand_over_nhwc(ctx, node, result)
        node.graph.erase_node(inp)
    if composed:
        node.graph.erase_node(source)


# -----------------------------------------------------------------------------
# add / sub
# -----------------------------------------------------------------------------


def _add_params(
    node: fx.Node, inp: QParams, other: QParams, out: QParams
) -> Optional[tuple[int, int, int, int, int, int]]:
    """quantized_add's multipliers and shifts, or None if they overflow.

    The second operand's multiplier carries alpha, and its sign the
    subtraction. A positive output shift takes arm_nn_requantize's left-shift
    branch, which can wrap in int32; how far is independent of how the scaling
    is split, so such an add is left to the portable kernels.
    """
    alpha = float(cast(float, get_arg(node, "alpha")))
    coefficient = -alpha if node.target == _fused_quant.sub.default else alpha
    other_scale = abs(coefficient) * other.scale
    max_scale_2x = 2 * max(inp.scale, other_scale)
    inp_multiplier, inp_shift = quantize_multiplier_aot(inp.scale / max_scale_2x)
    other_multiplier, other_shift = quantize_multiplier_aot(other_scale / max_scale_2x)
    if coefficient < 0:
        other_multiplier = -other_multiplier
    out_multiplier, out_shift = quantize_multiplier_aot(
        max_scale_2x / (out.scale * (1 << SHIFT_INT8))
    )
    if out_shift > 0:
        sign = -1 if coefficient < 0 else 1
        inp_span = (inp.quant_min - inp.zero_point, inp.quant_max - inp.zero_point)
        other_span = (
            other.quant_min - other.zero_point,
            other.quant_max - other.zero_point,
        )
        worst_case_sum = (
            max(
                abs(a * inp.scale + sign * b * other_scale)
                for a in inp_span
                for b in other_span
            )
            * (1 << SHIFT_INT8)
            / max_scale_2x
        )
        if worst_case_sum * 2**out_shift >= 2**31:
            return None
    return (
        inp_multiplier,
        inp_shift,
        other_multiplier,
        other_shift,
        out_multiplier,
        out_shift,
    )


def _can_lower_add(ctx: Context, node: fx.Node) -> bool:
    qparams = _binary_qparams(ctx, node)
    return (
        qparams is not None
        and _operand_layout(node) is not None
        and _add_params(node, *qparams) is not None
    )


def _lower_add(ctx: Context, node: fx.Node) -> None:
    qparams = _binary_qparams(ctx, node)
    assert qparams is not None
    inp, other, out = qparams
    params = _add_params(node, inp, other, out)
    assert params is not None
    inp_multiplier, inp_shift, other_multiplier, other_shift = params[:4]
    out_multiplier, out_shift = params[4:]
    _lower_binary(
        ctx,
        node,
        _cortex_m.quantized_add.default,
        lambda lhs, rhs: (
            lhs,
            inp.zero_point,
            inp_multiplier,
            inp_shift,
            rhs,
            other.zero_point,
            other_multiplier,
            other_shift,
            out.zero_point,
            out_multiplier,
            out_shift,
            out.quant_min,
            out.quant_max,
        ),
    )


# -----------------------------------------------------------------------------
# mul / div
# -----------------------------------------------------------------------------


def _can_lower_mul(ctx: Context, node: fx.Node) -> bool:
    qparams = _binary_qparams(ctx, node)
    return (
        qparams is not None
        and qparams[2].is_full_range
        and _operand_layout(node) is not None
    )


def _lower_mul(ctx: Context, node: fx.Node) -> None:
    qparams = _binary_qparams(ctx, node)
    assert qparams is not None
    inp, other, out = qparams
    multiplier, shift = quantize_multiplier_aot(inp.scale * other.scale / out.scale)
    _lower_binary(
        ctx,
        node,
        _cortex_m.quantized_mul.default,
        lambda lhs, rhs: (
            lhs,
            inp.zero_point,
            rhs,
            other.zero_point,
            out.zero_point,
            multiplier,
            shift,
        ),
    )


_DIV_DTYPES = (torch.int8, torch.int16)


def _can_lower_div(ctx: Context, node: fx.Node) -> bool:
    """The div kernel runs int8 or int16, on equal shapes."""
    qparams = _binary_qparams(ctx, node, _DIV_DTYPES)
    return (
        qparams is not None
        and len({qparams.dtype for qparams in qparams}) == 1
        and qparams[2].is_full_range
        and _same_shapes(node)
    )


def _lower_div(ctx: Context, node: fx.Node) -> None:
    """CMSIS-NN has no fixed-point divide; the kernel divides in float, so it
    takes the effective scale as is."""
    qparams = _binary_qparams(ctx, node, _DIV_DTYPES)
    assert qparams is not None
    inp, other, out = qparams
    ctx.replace(
        node,
        _cortex_m.quantized_div.default,
        (
            node.args[0],
            inp.zero_point,
            node.args[1],
            other.zero_point,
            out.zero_point,
            inp.scale / (other.scale * out.scale),
        ),
    )


# -----------------------------------------------------------------------------
# minimum / maximum
# -----------------------------------------------------------------------------


def _can_lower_min_max(ctx: Context, node: fx.Node) -> bool:
    """The kernels compare raw int8, so the three grids must be one."""
    qparams = _binary_qparams(ctx, node)
    return (
        qparams is not None
        and qparams[0] == qparams[1] == qparams[2]
        and qparams[2].is_full_range
        and len(node.meta["val"].shape) <= 4
    )


def _lower_min_max(ctx: Context, node: fx.Node) -> None:
    target = (
        _cortex_m.minimum.default
        if node.target == _fused_quant.minimum.default
        else _cortex_m.maximum.default
    )
    ctx.replace(node, target, (node.args[0], node.args[1]))


# -----------------------------------------------------------------------------
# softmax
# -----------------------------------------------------------------------------

_CMSIS_SOFTMAX_OUT = QParams(
    CMSIS_SOFTMAX_SCALE, CMSIS_SOFTMAX_ZERO_POINT, INT8_MIN, INT8_MAX
)


def _can_lower_softmax(ctx: Context, node: fx.Node) -> bool:
    """arm_softmax_s8 runs along the last dim; another dim runs transposed."""
    if _unary_qparams(ctx, node) is None or get_arg(node, "half_to_float"):
        return False
    rank = len(shape(node.args[0]))
    return get_arg(node, "dim", int) % rank == rank - 1 or rank <= 4


def _lower_softmax(ctx: Context, node: fx.Node) -> None:
    """arm_softmax_s8 writes a fixed grid, which the quantizer gives the output;
    quant absorption may have moved it to another, which a requantize table
    then reaches.

    Along another dim, the softmax runs on the tensor behind a permute that
    makes that dim the last one, or on a transpose.
    """
    qparams = _unary_qparams(ctx, node)
    assert qparams is not None
    inp, out = qparams
    x = cast(fx.Node, node.args[0])
    rank = len(shape(x))
    dim = get_arg(node, "dim", int) % rank
    permutation = None
    if dim != rank - 1:
        if x.target == _aten.permute_copy.default and (
            cast(list[int], x.args[1])[dim] % rank == rank - 1
        ):
            permutation = [d % rank for d in cast(list[int], x.args[1])]
            x = cast(fx.Node, x.args[0])
        else:
            permutation = list(range(rank))
            permutation[dim], permutation[-1] = rank - 1, dim
            x = ctx.call(node, _aten.permute_copy.default, (x, permutation))
    multiplier, shift, diff_min = _compute_softmax_params(inp.scale)
    result = ctx.call(
        node,
        _cortex_m.softmax.default,
        (
            x,
            -1,
            inp.zero_point,
            _CMSIS_SOFTMAX_OUT.zero_point,
            multiplier,
            shift,
            diff_min,
        ),
    )
    if permutation is not None:
        result = ctx.call(node, _aten.permute_copy.default, (result, permutation))
    if out != _CMSIS_SOFTMAX_OUT:
        result = ctx.call(
            node,
            _cortex_m.quantized_activation.default,
            (
                result,
                ctx.add_constant(
                    node, "requantize", requantize_table(_CMSIS_SOFTMAX_OUT, out)
                ),
            ),
        )
    bypassed = node.args[0]
    node.replace_all_uses_with(result)
    node.graph.erase_node(node)
    if bypassed is not x:
        _erase_if_unused(cast(fx.Node, bypassed))


# -----------------------------------------------------------------------------
# constant_pad_nd
# -----------------------------------------------------------------------------


def _read_as_nhwc(val: torch.Tensor) -> bool:
    """Whether the pad kernel reads a contiguous 4-d tensor as NHWC, as it does
    one of one channel or one pixel: an NHWC tensor one row high is one."""
    return (
        val.dim() == 4
        and is_channels_last(val)
        and tuple(val.stride()) == tuple(make_contiguous_strides_for(val.shape))
    )


def can_pad(val: torch.Tensor, pad: list[int]) -> bool:
    """Whether ``call_pad`` pads ``val`` by ATen-ordered ``pad``: one the
    kernel would read as NHWC pads with its first two dims merged, so the pad
    must leave dim 0, and dim 1 unless dim 0 is 1."""
    if not _read_as_nhwc(val):
        return True
    padded = list(pad) + [0] * (8 - len(pad))
    return not any(padded[6:]) and (val.shape[0] == 1 or not any(padded[4:6]))


def call_pad(
    ctx: Context, node: fx.Node, inp: fx.Node, pad: list[int], value: int
) -> fx.Node:
    """cortex_m.pad of ``inp`` by ATen-ordered ``pad``, ahead of node. A tensor
    the kernel would read as NHWC pads as 3-d, so the kernel and its meta keep
    the contiguous layout."""
    if not _read_as_nhwc(inp.meta["val"]):
        return ctx.call(node, _cortex_m.pad.default, pad_args(inp, pad, value))
    batch, channels, height, width = shape(inp)
    view = ctx.call(
        node, _aten.view_copy.default, (inp, [batch * channels, height, width])
    )
    padded = ctx.call(node, _cortex_m.pad.default, pad_args(view, pad[:6], value))
    merged, *spatial = shape(padded)
    return ctx.call(
        node, _aten.view_copy.default, (padded, [batch, merged // batch, *spatial])
    )


def _can_lower_pad(ctx: Context, node: fx.Node) -> bool:
    """arm_pad_s8 copies bytes, so the output keeps the input's grid; it pads
    up to four dims and does not crop."""
    qparams = _unary_qparams(ctx, node)
    pad = get_arg(node, "pad", list[int])
    rank = len(shape(node.args[0]))
    return (
        qparams is not None
        and qparams[0] == qparams[1]
        and 1 <= rank <= 4
        and len(pad) <= 2 * rank
        and all(isinstance(p, int) and p >= 0 for p in pad)
        and can_pad(cast(fx.Node, node.args[0]).meta["val"], pad)
    )


def pad_args(
    inp: fx.Node, pad: list[int], value: int
) -> tuple[fx.Node, list[int], list[int], int]:
    """cortex_m.pad's args for ATen-ordered ``pad``: last dim first, before and
    after. The kernel reads them in its view of the layout."""
    pre_pad, post_pad = [0] * 4, [0] * 4
    for i in range(len(pad) // 2):
        pre_pad[3 - i], post_pad[3 - i] = pad[2 * i], pad[2 * i + 1]
    val = inp.meta["val"]
    return (
        inp,
        to_physical_order(pre_pad, val),
        to_physical_order(post_pad, val),
        value,
    )


def _lower_pad(ctx: Context, node: fx.Node) -> None:
    qparams = _unary_qparams(ctx, node)
    assert qparams is not None
    out = qparams[1]
    value = int(
        quantize_val(
            float(cast(float, get_arg(node, "value"))),
            out.scale,
            out.zero_point,
            out.quant_min,
            out.quant_max,
        )
    )
    inp = get_arg(node, "inp", fx.Node)
    node.replace_all_uses_with(
        call_pad(ctx, node, inp, get_arg(node, "pad", list[int]), value)
    )
    node.graph.erase_node(node)


LUT_TARGETS = (
    _fused_quant.relu.default,
    _fused_quant.hard_tanh.default,
    _fused_quant.sigmoid.default,
    _fused_quant.tanh.default,
    _fused_quant.gelu.default,
    _fused_quant.hardswish.default,
    _fused_quant.hardsigmoid.default,
    _fused_quant.silu.default,
    _fused_quant.clamp.default,
    _fused_quant.leaky_relu.default,
    _fused_quant.log.default,
    _fused_quant.log2.default,
    _fused_quant.log10.default,
    _fused_quant.log1p.default,
    _fused_quant.sqrt.default,
    _fused_quant.rsqrt.default,
    _fused_quant.pow.Tensor_Scalar,
    _fused_quant.requantize.default,
    _fused_quant.add.Scalar,
    _fused_quant.sub.Scalar,
    _fused_quant.mul.Scalar,
    _fused_quant.div.Scalar,
)

RULES: Rules = {
    **{target: Rule(_can_lower_lut, _lower_lut) for target in LUT_TARGETS},
    _fused_quant.add.default: Rule(_can_lower_add, _lower_add),
    _fused_quant.sub.default: Rule(_can_lower_add, _lower_add),
    _fused_quant.mul.default: Rule(_can_lower_mul, _lower_mul),
    _fused_quant.div.default: Rule(_can_lower_div, _lower_div),
    _fused_quant.minimum.default: Rule(_can_lower_min_max, _lower_min_max),
    _fused_quant.maximum.default: Rule(_can_lower_min_max, _lower_min_max),
    _fused_quant._softmax.default: Rule(_can_lower_softmax, _lower_softmax),
    _fused_quant.constant_pad_nd.default: Rule(_can_lower_pad, _lower_pad),
}
