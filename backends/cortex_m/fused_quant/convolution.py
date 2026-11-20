# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lowerings of the channels-last fused_quant convolution and pools.

``ToChannelsLast`` gives these ops NHWC activations and OHWI weights, the
logical layout of CMSIS-NN's ``*_nhwc`` kernels, so they lower without a
transpose of their own.
"""

import operator
from collections.abc import Sequence
from typing import cast, Optional

import torch
from executorch.backends.cortex_m.fused_quant.elementwise import call_pad, can_pad
from executorch.backends.cortex_m.fused_quant.utils import (
    Context,
    requantize_params,
    requantize_table,
    Rule,
    Rules,
    shape,
)
from executorch.backends.cortex_m.library import cmsis_nn
from executorch.backends.cortex_m.utils import is_channels_last, quantize_multiplier_aot
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.exir.dialects._ops import ops as exir_ops
from torch import fx
from torch.fx.node import Argument

_aten = exir_ops.edge.aten
_cortex_m = exir_ops.edge.cortex_m
_fused_quant = exir_ops.edge.fused_quant

# The CMSIS-NN repacking threshold below which MVE runs a depthwise
# convolution of a single input channel as a regular one.
_MVE_DEPTHWISE_MAX_OUT_CHANNELS = 8

_NCHW_TO_NHWC = [0, 2, 3, 1]
_NHWC_TO_NCHW = [0, 3, 1, 2]


def _pair(values: Sequence[int]) -> list[int]:
    return [values[0]] * 2 if len(values) == 1 else list(values)


def _ints(node: fx.Node, name: str) -> list[int]:
    return list(cast(Sequence[int] | None, get_arg(node, name)) or ())


def _out_channel_axis(node: fx.Node) -> int:
    """A transposed weight is [C_in, *spatial, C_out] channels-last."""
    if not get_arg(node, "transposed", bool):
        return 0
    return len(shape(get_arg(node, "weight"))) - 1


def _conv_weight_scales(ctx: Context, node: fx.Node) -> torch.Tensor | None:
    axis = _out_channel_axis(node)
    channels = shape(get_arg(node, "weight"))[axis]
    return ctx.weight_scales(node, axis, channels)


def _can_lower_conv(ctx: Context, node: fx.Node) -> bool:
    spatial_dims = len(_ints(node, "stride"))
    if (
        spatial_dims not in (1, 2)
        or ctx.qparams(node, "inp") is None
        or ctx.qparams(node, "out") is None
        or _conv_weight_scales(ctx, node) is None
        or not ctx.has_constant_bias(node)
    ):
        return False
    if not get_arg(node, "transposed", bool):
        return True
    # arm_transpose_conv_s8 has no groups, dilation or output padding.
    return (
        spatial_dims == 2
        and get_arg(node, "groups", int) == 1
        and all(d == 1 for d in _ints(node, "dilation"))
        and all(p == 0 for p in _ints(node, "output_padding"))
    )


def _is_depthwise(ctx: Context, node: fx.Node, weight: torch.Tensor) -> bool:
    """Whether to run arm_depthwise_conv_wrapper_s8 rather than a grouped
    arm_convolve_wrapper_s8.

    Only for a batch of one, where CMSIS-NN has an optimized depthwise kernel.
    An MVE core already repacks a single-channel depthwise weight and runs it
    as a regular convolution; emitting that layout saves the repacking.
    """
    groups = get_arg(node, "groups", int)
    in_channels = weight.shape[-1] * groups
    out_channels = weight.shape[0]
    if (
        groups != in_channels
        or out_channels % in_channels
        or node.meta["val"].shape[0] != 1
    ):
        return False
    return not (
        ctx.target_config.backend == cmsis_nn.Backend.MVE
        and in_channels == 1
        and out_channels > _MVE_DEPTHWISE_MAX_OUT_CHANNELS
    )


def _logical_pad(pad: fx.Node) -> tuple[fx.Node, list[int], list[int]]:
    """A 4-d cortex_m.pad's source and padding, before and after each dim,
    in the source's logical dim order."""
    source, before, after, _ = cast(tuple[fx.Node, list[int], list[int], int], pad.args)
    if is_channels_last(source.meta["val"]):
        return (
            source,
            [before[i] for i in _NHWC_TO_NCHW],
            [after[i] for i in _NHWC_TO_NCHW],
        )
    return source, list(before), list(after)


def _fold_padding(
    ctx: Context,
    node: fx.Node,
    x: fx.Node,
    kernel: Sequence[int],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
) -> tuple[fx.Node, list[int], list[fx.Node]]:
    """The input and padding of a 2-d convolution, with an explicit pad of
    the input zero point ahead of it folded in, as the current flow's
    FuseConvPaddingPass folds it, and the nodes the fold leaves unread.

    The kernels take padding as top, left, bottom and right, but only SAME's:
    the bottom and right padding are those the output size implies. The pad
    pads the NHWC input, or the NCHW tensor behind the permute to NHWC.
    """
    inp = ctx.qparams(node, "inp")
    assert inp is not None
    unchanged: tuple[fx.Node, list[int], list[fx.Node]] = (x, padding, [])
    permute = None
    pad = x
    if (
        x.target == _aten.permute_copy.default
        and list(cast(Sequence[int], x.args[1])) == _NCHW_TO_NHWC
        and len(x.users) == 1
    ):
        permute, pad = x, cast(fx.Node, x.args[0])
    if (
        pad.target != _cortex_m.pad.default
        or len(pad.users) != 1
        or len(shape(pad)) != 4
        or pad.args[3] != inp.zero_point
    ):
        return unchanged
    source, before, after = _logical_pad(pad)
    height_dim, width_dim, channel_dim = (2, 3, 1) if permute is not None else (1, 2, 3)
    if any(before[d] or after[d] for d in (0, channel_dim)):
        return unchanged
    size = (shape(source)[height_dim], shape(source)[width_dim])
    # The pinned CMSIS-NN MVE 1xN kernel mishandles asymmetric padding.
    if size[0] == 1 and kernel[0] == 1:
        return unchanged
    fused = [
        before[height_dim] + padding[0],
        before[width_dim] + padding[1],
        after[height_dim] + padding[0],
        after[width_dim] + padding[1],
    ]
    total = [
        max(((length + step - 1) // step - 1) * step + dil * (k - 1) + 1 - length, 0)
        for length, step, dil, k in zip(size, stride, dilation, kernel)
    ]
    if fused != [
        total[0] // 2,
        total[1] // 2,
        total[0] - total[0] // 2,
        total[1] - total[1] // 2,
    ]:
        return unchanged
    if permute is None:
        return source, fused, [pad]
    nhwc = ctx.call(node, _aten.permute_copy.default, (source, _NCHW_TO_NHWC))
    return nhwc, fused, [permute, pad]


def _lower_conv(ctx: Context, node: fx.Node) -> None:
    """A 1-d convolution runs as a 2-d one over a height of one."""
    inp, out = ctx.qparams(node, "inp"), ctx.qparams(node, "out")
    scales = _conv_weight_scales(ctx, node)
    weight_node: fx.Node | None = get_arg(node, "weight", fx.Node)
    weight = ctx.constant(weight_node)
    assert inp is not None and out is not None
    assert scales is not None and weight is not None
    stride, padding = _ints(node, "stride"), _ints(node, "padding")
    dilation, groups = _ints(node, "dilation"), get_arg(node, "groups", int)
    transposed = get_arg(node, "transposed", bool)

    x = get_arg(node, "inp", fx.Node)
    unread: list[fx.Node] = []
    if len(stride) == 1:
        batch, length, channels = shape(x)
        x = ctx.call(node, _aten.view_copy.default, (x, [batch, 1, length, channels]))
        weight = weight.unsqueeze(1)
        weight_node = None
        stride, padding, dilation = [1, *stride], [0, *padding], [1, *dilation]
    elif not transposed:
        x, padding, unread = _fold_padding(
            ctx, node, x, weight.shape[1:3], stride, padding, dilation
        )

    bias = ctx.int32_bias(node, inp.scale, scales)
    if bias is None and groups > 1:
        # The grouped and MVE depthwise kernels offset the bias pointer per
        # group or channel block, so it cannot be null.
        bias = torch.zeros(scales.numel(), dtype=torch.int32)
    bias_arg = None if bias is None else ctx.add_constant(node, "bias", bias)
    multipliers, shifts = requantize_params(inp.scale, scales, out.scale)
    requantize_args = (
        -inp.zero_point,
        out.zero_point,
        ctx.add_constant(
            node, "multipliers", torch.tensor(multipliers, dtype=torch.int32)
        ),
        ctx.add_constant(node, "shifts", torch.tensor(shifts, dtype=torch.int32)),
        out.quant_min,
        out.quant_max,
    )

    def weight_arg(tensor: torch.Tensor) -> fx.Node:
        return ctx.add_constant(node, "weight", tensor.contiguous())

    if transposed:
        # [C_in, H, W, C_out] -> [C_out, H, W, C_in]
        target = _cortex_m.quantized_transpose_conv2d_nhwc.default
        args: tuple[Argument, ...] = (
            x,
            weight_arg(weight.permute(3, 1, 2, 0)),
            bias_arg,
            stride,
            padding,
            _ints(node, "output_padding"),
            dilation,
            *requantize_args,
            ctx.scratch(node),
            ctx.scratch(node),
        )
    elif _is_depthwise(ctx, node, weight):
        # [C_out, H, W, 1] -> [1, H, W, C_out]
        target = _cortex_m.quantized_depthwise_conv2d_nhwc.default
        args = (
            x,
            weight_arg(weight.permute(3, 1, 2, 0)),
            bias_arg,
            stride,
            padding,
            dilation,
            weight.shape[0] // groups,
            *requantize_args,
            ctx.scratch(node),
        )
    else:
        target = _cortex_m.quantized_conv2d_nhwc.default
        args = (
            x,
            weight_node if weight_node is not None else weight_arg(weight),
            bias_arg,
            stride,
            padding,
            dilation,
            *requantize_args,
            ctx.scratch(node),
        )

    conv = ctx.call(node, target, args)
    if len(_ints(node, "stride")) == 1:
        conv = ctx.call(
            node, _aten.view_copy.default, (conv, list(node.meta["val"].shape))
        )
    node.replace_all_uses_with(conv)
    node.graph.erase_node(node)
    for unread_node in unread:
        node.graph.erase_node(unread_node)


def _same_grid(ctx: Context, node: fx.Node) -> bool:
    inp, out = ctx.qparams(node, "inp"), ctx.qparams(node, "out")
    return (
        inp is not None
        and out is not None
        and (inp.scale, inp.zero_point) == (out.scale, out.zero_point)
    )


def _can_lower_max_pool(ctx: Context, node: fx.Node) -> bool:
    return (
        _same_grid(ctx, node)
        and _pair(_ints(node, "dilation")) == [1, 1]
        and not get_arg(node, "ceil_mode", bool)
        and all(
            user.target is operator.getitem and user.args[1] == 0 for user in node.users
        )
    )


def _lower_max_pool(ctx: Context, node: fx.Node) -> None:
    """arm_max_pool_s8 takes the values only; the indices have no users."""
    inp, out = ctx.qparams(node, "inp"), ctx.qparams(node, "out")
    assert inp is not None and out is not None
    kernel_size = _pair(_ints(node, "kernel_size"))
    pool = ctx.call(
        node,
        _cortex_m.quantized_max_pool2d_nhwc.default,
        (
            get_arg(node, "inp", fx.Node),
            kernel_size,
            _pair(_ints(node, "stride") or kernel_size),
            _pair(_ints(node, "padding")),
            [1, 1],
            False,
            inp.zero_point,
            out.zero_point,
            out.quant_min,
            out.quant_max,
        ),
    )
    for user in list(node.users):
        user.replace_all_uses_with(pool)
        node.graph.erase_node(user)
    node.graph.erase_node(node)


def _avg_pool_pad(node: fx.Node) -> Optional[list[int]]:
    """The explicit ATen-ordered pad of a pool that counts its padding:
    arm_avgpool_s8 leaves padding out of the count."""
    padding = _pair(_ints(node, "padding"))
    if not (get_arg(node, "count_include_pad", bool) and any(padding)):
        return None
    pad_h, pad_w = padding
    return [0, 0, pad_w, pad_w, pad_h, pad_h]


def _can_lower_avg_pool(ctx: Context, node: fx.Node) -> bool:
    """arm_avgpool_s8 writes on its input's grid and divides by the window."""
    out = ctx.qparams(node, "out")
    pad = _avg_pool_pad(node)
    return (
        _same_grid(ctx, node)
        and out is not None
        and out.is_full_range
        and get_arg(node, "divisor_override") is None
        and (pad is None or can_pad(get_arg(node, "inp", fx.Node).meta["val"], pad))
    )


def _lower_avg_pool(ctx: Context, node: fx.Node) -> None:
    """A pool that counts its padding pads explicitly, with the zero point."""
    inp = ctx.qparams(node, "inp")
    assert inp is not None
    kernel_size = _pair(_ints(node, "kernel_size"))
    padding = _pair(_ints(node, "padding"))
    x = get_arg(node, "inp", fx.Node)
    pad = _avg_pool_pad(node)
    if pad is not None:
        x = call_pad(ctx, node, x, pad, inp.zero_point)
        padding = [0, 0]
    multiplier, shift = quantize_multiplier_aot(inp.scale)
    ctx.replace(
        node,
        _cortex_m.quantized_avg_pool2d_nhwc.default,
        (
            x,
            kernel_size,
            _pair(_ints(node, "stride") or kernel_size),
            padding,
            get_arg(node, "ceil_mode", bool),
            inp.zero_point,
            multiplier,
            shift,
            ctx.scratch(node),
        ),
    )


def _can_lower_mean(ctx: Context, node: fx.Node) -> bool:
    """A mean over H and W is an average pool over the whole image."""
    inp, out = ctx.qparams(node, "inp"), ctx.qparams(node, "out")
    dims: Argument = get_arg(node, "dim")
    return (
        inp is not None
        and out is not None
        and out.is_full_range
        and len(shape(get_arg(node, "inp"))) == 4
        and isinstance(dims, (list, tuple))
        and sorted(cast(int, d) % 4 for d in dims) == [2, 3]
        and get_arg(node, "dtype") is None
    )


def _lower_mean(ctx: Context, node: fx.Node) -> None:
    """arm_avgpool_s8 pools NHWC, so the mean takes the NHWC tensor behind the
    permute that made its input NCHW, or permutes it. The pool writes on its
    input's grid; quant absorption may have moved the output to another, which
    a requantize table then reaches."""
    inp, out = ctx.qparams(node, "inp"), ctx.qparams(node, "out")
    assert inp is not None and out is not None
    x = get_arg(node, "inp", fx.Node)
    bypassed = (
        x.target == _aten.permute_copy.default
        and list(cast(Sequence[int], x.args[1])) == _NHWC_TO_NCHW
    )
    nhwc = (
        cast(fx.Node, x.args[0])
        if bypassed
        else ctx.call(node, _aten.permute_copy.default, (x, [0, 2, 3, 1]))
    )
    window = list(shape(nhwc)[1:3])
    multiplier, shift = quantize_multiplier_aot(inp.scale)
    pool = ctx.call(
        node,
        _cortex_m.quantized_avg_pool2d_nhwc.default,
        (
            nhwc,
            window,
            window,
            [0, 0],
            False,
            inp.zero_point,
            multiplier,
            shift,
            ctx.scratch(node),
        ),
    )
    # [N, 1, 1, C] holds the bytes of [N, C, 1, 1] and [N, C] alike.
    result = ctx.call(
        node, _aten.view_copy.default, (pool, list(node.meta["val"].shape))
    )
    if (inp.scale, inp.zero_point) != (out.scale, out.zero_point):
        result = ctx.call(
            node,
            _cortex_m.quantized_activation.default,
            (result, ctx.add_constant(node, "requantize", requantize_table(inp, out))),
        )
    node.replace_all_uses_with(result)
    node.graph.erase_node(node)
    if bypassed and not x.users:
        node.graph.erase_node(x)


RULES: Rules = {
    _fused_quant.convolution_channels_last.default: Rule(_can_lower_conv, _lower_conv),
    _fused_quant.max_pool2d_with_indices_channels_last.default: Rule(
        _can_lower_max_pool, _lower_max_pool
    ),
    _fused_quant.avg_pool2d_channels_last.default: Rule(
        _can_lower_avg_pool, _lower_avg_pool
    ),
    _fused_quant.mean.dim: Rule(_can_lower_mean, _lower_mean),
}
