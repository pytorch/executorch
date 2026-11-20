# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lowerings of fused_quant linear and bmm."""

from typing import cast

import torch
from executorch.backends.cortex_m.fused_quant.utils import (
    Context,
    requantize_params,
    Rule,
    Rules,
    shape,
)
from executorch.backends.cortex_m.library import cmsis_nn
from executorch.backends.cortex_m.passes.aten_to_cortex_m_pass import (
    _compute_kernel_sum,
)
from executorch.backends.cortex_m.utils import quantize_multiplier_aot
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.exir.dialects._ops import ops as exir_ops
from torch import fx
from torch.fx.node import Argument

_cortex_m = exir_ops.edge.cortex_m
_fused_quant = exir_ops.edge.fused_quant


def _can_lower_linear(ctx: Context, node: fx.Node) -> bool:
    weight: Argument = get_arg(node, "weight")
    return (
        ctx.qparams(node, "inp") is not None
        and ctx.qparams(node, "out") is not None
        and len(shape(weight)) == 2
        and ctx.weight_scales(node, axis=0, channels=shape(weight)[0]) is not None
        and ctx.has_constant_bias(node)
    )


def _lower_linear(ctx: Context, node: fx.Node) -> None:
    """arm_fully_connected_s8, per tensor or per channel.

    The MVE kernel reads a precomputed kernel sum, input_offset * sum(w) + bias,
    and ignores the bias argument; the DSP and scalar kernels read the bias and
    ignore the kernel sum.
    """
    inp, out = ctx.qparams(node, "inp"), ctx.qparams(node, "out")
    assert inp is not None and out is not None
    weight_node = cast(fx.Node, get_arg(node, "weight"))
    weight = ctx.constant(weight_node)
    assert weight is not None
    scales = ctx.weight_scales(node, axis=0, channels=weight.shape[0])
    assert scales is not None
    bias = ctx.int32_bias(node, inp.scale, scales)
    multipliers, shifts = requantize_params(inp.scale, scales, out.scale)

    kernel_sum_arg = bias_arg = None
    if ctx.target_config.backend == cmsis_nn.Backend.MVE:
        kernel_sum_arg = ctx.add_constant(
            node, "kernel_sum", _compute_kernel_sum(weight, bias, -inp.zero_point, 0)
        )
    elif bias is not None:
        bias_arg = ctx.add_constant(node, "bias", bias)

    if len(set(zip(multipliers, shifts))) == 1:
        target = _cortex_m.quantized_linear.default
        multipliers_arg: Argument = multipliers[:1]
        shifts_arg: Argument = shifts[:1]
    else:
        target = _cortex_m.quantized_linear_per_channel.default
        multipliers_arg = ctx.add_constant(
            node, "multipliers", torch.tensor(multipliers, dtype=torch.int32)
        )
        shifts_arg = ctx.add_constant(
            node, "shifts", torch.tensor(shifts, dtype=torch.int32)
        )
    ctx.replace(
        node,
        target,
        (
            node.args[0],
            weight_node,
            bias_arg,
            kernel_sum_arg,
            -inp.zero_point,
            0,
            out.zero_point,
            multipliers_arg,
            shifts_arg,
            out.quant_max,
            out.quant_min,
        ),
    )


def _can_lower_bmm(ctx: Context, node: fx.Node) -> bool:
    out = ctx.qparams(node, "out")
    return (
        ctx.qparams(node, "inp") is not None
        and ctx.qparams(node, "other") is not None
        and out is not None
        and out.is_full_range
        and len(shape(node.args[0])) == len(shape(node.args[1])) == 3
    )


def _lower_bmm(ctx: Context, node: fx.Node) -> None:
    """arm_batch_matmul_s8 takes the right-hand side transposed."""
    lhs, rhs = ctx.qparams(node, "inp"), ctx.qparams(node, "other")
    out = ctx.qparams(node, "out")
    assert lhs is not None and rhs is not None and out is not None
    multiplier, shift = quantize_multiplier_aot(lhs.scale * rhs.scale / out.scale)
    rhs_node = cast(fx.Node, node.args[1])
    rhs_constant = ctx.constant(rhs_node)
    if rhs_constant is not None:
        rhs_transposed = ctx.add_constant(
            node, "rhs_transposed", rhs_constant.permute(0, 2, 1).contiguous()
        )
    else:
        rhs_transposed = ctx.call(
            node, _cortex_m.transpose.default, (rhs_node, [0, 2, 1])
        )
    ctx.replace(
        node,
        _cortex_m.quantized_batch_matmul.default,
        (
            node.args[0],
            -lhs.zero_point,
            rhs_transposed,
            -rhs.zero_point,
            out.zero_point,
            multiplier,
            shift,
            ctx.scratch(node),
        ),
    )


RULES: Rules = {
    _fused_quant.linear.default: Rule(_can_lower_linear, _lower_linear),
    _fused_quant.bmm.default: Rule(_can_lower_bmm, _lower_bmm),
}
