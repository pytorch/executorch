# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast, Optional

import torch
from executorch.backends.fused_quant.graph_utils import (
    get_constant,
    get_qparams_from_node,
    set_constant,
)
from executorch.backends.fused_quant.ops import QuantParamsStruct
from executorch.backends.transforms.permute_pass_utils import get_arg, set_arg
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from executorch.exir.passes.constant_prop_pass import constant_prop_pass
from torch import fx
from torch.export import ExportedProgram

logger: logging.Logger = logging.getLogger(__name__)

_PASSTHROUGH_TARGETS: set[EdgeOpOverload] = {
    exir_ops.edge.aten.permute_copy.default,
    exir_ops.edge.aten.view_copy.default,
}
_CONV_TARGETS: set[EdgeOpOverload] = {
    exir_ops.edge.fused_quant.convolution.default,
    exir_ops.edge.fused_quant.convolution_channels_last.default,
}
_AFFINE_TARGETS: set[EdgeOpOverload] = {
    exir_ops.edge.fused_quant.linear.default,
    *_CONV_TARGETS,
}


@dataclass(frozen=True)
class _ConstantUpdate:
    node: fx.Node
    value: torch.Tensor


@dataclass(frozen=True)
class _SignedWeightPlan:
    weight_update: Optional[_ConstantUpdate]
    zero_point_update: Optional[_ConstantUpdate]


@dataclass(frozen=True)
class _FoldPlan:
    affine_node: fx.Node
    passthrough_chain: list[fx.Node]
    weight_scale_node: fx.Node
    new_weight_scale: torch.Tensor
    signed_weight_plan: _SignedWeightPlan
    bias_node: Optional[fx.Node]
    new_bias: Optional[torch.Tensor]


def _find_affine_through_passthrough(
    node: fx.Node,
) -> Optional[tuple[fx.Node, list[fx.Node]]]:
    """Walk backward from *node* through passthrough ops to find an affine op.

    Returns ``(affine_node, passthrough_chain)`` where *passthrough_chain* is the
    list of passthrough nodes between the affine and *node* (empty when *node* is
    the affine itself). Returns ``None`` if no affine is reachable through
    single-user passthrough ops.
    """

    chain: list[fx.Node] = []
    current = node
    while current.target in _PASSTHROUGH_TARGETS:
        if len(current.users) != 1:
            return None
        chain.append(current)
        prev = current.args[0]
        if not isinstance(prev, fx.Node):
            return None
        current = prev

    if current.target not in _AFFINE_TARGETS:
        return None
    if current.target in _CONV_TARGETS and get_arg(current, "transposed", bool):
        return None

    affine_node = current
    if len(affine_node.users) != 1:
        return None

    return affine_node, chain


def _affine_output_channel_axis(affine_node: fx.Node) -> int:
    if affine_node.target == exir_ops.edge.fused_quant.convolution.default:
        return 1
    return affine_node.meta["val"].ndim - 1


def _channel_axis_after_view(
    input_shape: torch.Size,
    output_shape: torch.Size,
    input_channel_axis: int,
) -> Optional[int]:
    channel_size = input_shape[input_channel_axis]
    prefix_size = math.prod(input_shape[:input_channel_axis])
    suffix_size = math.prod(input_shape[input_channel_axis + 1 :])
    candidates = [
        axis
        for axis, size in enumerate(output_shape)
        if size == channel_size
        and math.prod(output_shape[:axis]) == prefix_size
        and math.prod(output_shape[axis + 1 :]) == suffix_size
    ]
    return candidates[0] if len(candidates) == 1 else None


def _channel_axis_after_passthrough(
    affine_node: fx.Node,
    passthrough_chain: list[fx.Node],
) -> Optional[int]:
    channel_axis = _affine_output_channel_axis(affine_node)
    input_shape = affine_node.meta["val"].shape
    for passthrough in reversed(passthrough_chain):
        output_shape = passthrough.meta["val"].shape
        if passthrough.target == exir_ops.edge.aten.permute_copy.default:
            dims = [
                dim % len(input_shape)
                for dim in get_arg(passthrough, "dims", list[int])
            ]
            if len(dims) != len(input_shape) or sorted(dims) != list(
                range(len(input_shape))
            ):
                return None
            channel_axis = dims.index(channel_axis)
        else:
            view_channel_axis = _channel_axis_after_view(
                input_shape,
                output_shape,
                channel_axis,
            )
            if view_channel_axis is None:
                return None
            channel_axis = view_channel_axis
        input_shape = output_shape
    return channel_axis


def _constant_to_channel_scale(
    scale_tensor: torch.Tensor,
    output_shape: torch.Size,
    output_channel_axis: Optional[int],
) -> Optional[torch.Tensor]:
    """Collapse a scalar or output-channel multiplier to one dimension."""
    if scale_tensor.ndim > len(output_shape):
        return None

    aligned_shape = (1,) * (len(output_shape) - scale_tensor.ndim) + tuple(
        scale_tensor.shape
    )
    if any(
        scale_size not in (1, output_size)
        for scale_size, output_size in zip(aligned_shape, output_shape)
    ):
        return None

    channel_scale = scale_tensor.reshape(-1)
    if channel_scale.is_complex() or not bool(torch.all(torch.isfinite(channel_scale))):
        return None
    if channel_scale.numel() == 1:
        return channel_scale
    if output_channel_axis is None:
        return None
    if any(
        axis != output_channel_axis and size != 1
        for axis, size in enumerate(aligned_shape)
    ):
        return None
    if aligned_shape[output_channel_axis] != output_shape[output_channel_axis]:
        return None
    return channel_scale


def _is_per_channel_qparam(qparam: torch.Tensor, out_channels: int) -> bool:
    return (
        qparam.numel() == out_channels
        and qparam.shape[0] == out_channels
        and all(size == 1 for size in qparam.shape[1:])
    )


def _reshape_channel_values(
    channel_values: torch.Tensor,
    ndim: int,
    out_channels: int,
) -> Optional[torch.Tensor]:
    if channel_values.numel() == 1:
        return channel_values.reshape(())
    if channel_values.numel() != out_channels or ndim == 0:
        return None
    return channel_values.reshape([out_channels] + [1] * (ndim - 1))


def _get_weight_scale_factor(
    weight_scale: torch.Tensor,
    channel_scale: torch.Tensor,
    out_channels: int,
) -> Optional[torch.Tensor]:
    if channel_scale.numel() != 1 and not _is_per_channel_qparam(
        weight_scale, out_channels
    ):
        return None
    return _reshape_channel_values(channel_scale, weight_scale.ndim, out_channels)


def _get_new_weight_scale(
    weight_scale: torch.Tensor,
    channel_scale: torch.Tensor,
    out_channels: int,
) -> Optional[torch.Tensor]:
    scale_factor = _get_weight_scale_factor(
        weight_scale,
        channel_scale.abs(),
        out_channels,
    )
    zero_mask = _get_weight_scale_factor(
        weight_scale,
        channel_scale == 0,
        out_channels,
    )
    if scale_factor is None or zero_mask is None:
        return None
    scale_factor = scale_factor.to(weight_scale)
    zero_mask = zero_mask.to(device=weight_scale.device)
    return torch.where(
        zero_mask,
        torch.ones_like(weight_scale),
        weight_scale * scale_factor,
    )


def _update_integer_channels(
    value: torch.Tensor,
    channel_scale: torch.Tensor,
    out_channels: int,
    quantized_range_sum: int,
) -> Optional[torch.Tensor]:
    if value.is_floating_point() or value.is_complex():
        return None
    shaped_scale = _reshape_channel_values(
        channel_scale,
        value.ndim,
        out_channels,
    )
    if shaped_scale is None:
        return None
    shaped_scale = shaped_scale.to(device=value.device)
    # Applying the same range reflection to q and z gives q' - z' = -(q - z).
    reflected = (quantized_range_sum - value.to(torch.int64)).to(value.dtype)
    updated = torch.where(shaped_scale < 0, reflected, value)
    return torch.where(shaped_scale == 0, torch.zeros_like(updated), updated)


def _get_signed_weight_plan(
    exported_program: ExportedProgram,
    affine_node: fx.Node,
    channel_scale: torch.Tensor,
    out_channels: int,
) -> Optional[_SignedWeightPlan]:
    if bool(torch.all(channel_scale > 0)):
        return _SignedWeightPlan(None, None)

    weight_node = get_arg(affine_node, "weight", fx.Node)
    zero_point_node = get_arg(affine_node, "weight_zero_point", fx.Node)
    if len(weight_node.users) != 1 or len(zero_point_node.users) != 1:
        return None
    weight = get_constant(exported_program, weight_node)
    zero_point = get_constant(exported_program, zero_point_node)
    if weight is None or zero_point is None:
        return None
    if weight.ndim == 0 or weight.shape[0] != out_channels:
        return None
    if channel_scale.numel() != 1 and not _is_per_channel_qparam(
        zero_point, out_channels
    ):
        return None

    quant_min = get_arg(affine_node, "weight_quant_min", int)
    quant_max = get_arg(affine_node, "weight_quant_max", int)
    if bool(torch.any(channel_scale == 0)) and not quant_min <= 0 <= quant_max:
        return None
    range_sum = quant_min + quant_max
    new_weight = _update_integer_channels(
        weight, channel_scale, out_channels, range_sum
    )
    new_zero_point = _update_integer_channels(
        zero_point, channel_scale, out_channels, range_sum
    )
    if new_weight is None or new_zero_point is None:
        return None
    return _SignedWeightPlan(
        _ConstantUpdate(weight_node, new_weight),
        _ConstantUpdate(zero_point_node, new_zero_point),
    )


def _get_scaled_bias(
    exported_program: ExportedProgram,
    affine_node: fx.Node,
    channel_scale: torch.Tensor,
) -> Optional[tuple[Optional[fx.Node], Optional[torch.Tensor]]]:
    bias_node = get_arg(affine_node, "bias", Optional[fx.Node])
    if bias_node is None:
        return None, None
    if len(bias_node.users) != 1:
        return None
    if get_arg(affine_node, "bias_scale", Optional[fx.Node]) is not None:
        return None
    bias = get_constant(exported_program, bias_node)
    if bias is None:
        return None
    return bias_node, bias * channel_scale.to(bias)


def _make_fold_plan(
    exported_program: ExportedProgram,
    data_node: fx.Node,
    scale_tensor: torch.Tensor,
) -> Optional[_FoldPlan]:
    affine_info = _find_affine_through_passthrough(data_node)
    if affine_info is None:
        return None
    affine_node, passthrough_chain = affine_info

    weight_scale_node = get_arg(affine_node, "weight_scale", Optional[fx.Node])
    if weight_scale_node is None or len(weight_scale_node.users) != 1:
        return None
    weight_scale = get_constant(exported_program, weight_scale_node)
    if weight_scale is None:
        return None

    output_channel_axis = None
    if scale_tensor.numel() != 1:
        output_channel_axis = _channel_axis_after_passthrough(
            affine_node,
            passthrough_chain,
        )
    channel_scale = _constant_to_channel_scale(
        scale_tensor,
        data_node.meta["val"].shape,
        output_channel_axis,
    )
    if channel_scale is None:
        return None

    affine_channel_axis = _affine_output_channel_axis(affine_node)
    out_channels = cast(torch.Tensor, affine_node.meta["val"]).shape[
        affine_channel_axis
    ]
    new_weight_scale = _get_new_weight_scale(
        weight_scale,
        channel_scale,
        out_channels,
    )
    if new_weight_scale is None:
        return None

    signed_weight_plan = _get_signed_weight_plan(
        exported_program,
        affine_node,
        channel_scale,
        out_channels,
    )
    if signed_weight_plan is None:
        return None

    scaled_bias = _get_scaled_bias(exported_program, affine_node, channel_scale)
    if scaled_bias is None:
        return None
    bias_node, new_bias = scaled_bias
    return _FoldPlan(
        affine_node,
        passthrough_chain,
        weight_scale_node,
        new_weight_scale,
        signed_weight_plan,
        bias_node,
        new_bias,
    )


def _apply_fold(
    exported_program: ExportedProgram,
    mul_node: fx.Node,
    data_node: fx.Node,
    plan: _FoldPlan,
) -> None:
    set_constant(
        exported_program,
        plan.weight_scale_node,
        plan.new_weight_scale,
    )
    if plan.signed_weight_plan.weight_update is not None:
        set_constant(
            exported_program,
            plan.signed_weight_plan.weight_update.node,
            plan.signed_weight_plan.weight_update.value,
        )
    if plan.signed_weight_plan.zero_point_update is not None:
        set_constant(
            exported_program,
            plan.signed_weight_plan.zero_point_update.node,
            plan.signed_weight_plan.zero_point_update.value,
        )
    if plan.bias_node is not None and plan.new_bias is not None:
        set_constant(exported_program, plan.bias_node, plan.new_bias)

    for field in (
        "out_zero_point",
        "out_scale",
        "out_dtype",
        "out_quant_min",
        "out_quant_max",
    ):
        set_arg(plan.affine_node, field, get_arg(mul_node, field))

    new_dtype = mul_node.meta["val"].dtype
    for retyped in (plan.affine_node, *plan.passthrough_chain):
        val = cast(torch.Tensor, retyped.meta["val"])
        retyped.meta["val"] = val.to(new_dtype)

    mul_node.replace_all_uses_with(data_node)
    mul_node.graph.erase_node(mul_node)


def _trace_to_constant(
    ep: ExportedProgram,
    node: fx.Node,
) -> Optional[tuple[torch.Tensor, fx.Node]]:
    """Find the constant tensor backing a node.

    Handles:
    - Direct placeholder node → returns its stored tensor, which may be
      floating point or integer quantized storage when an upstream pass folded
      the quantize node.
    - quantize_per_tensor(placeholder, ...) → returns the floating-point source
      constant and the quantize node.

    Callers must use the consuming op's qparams to interpret integer storage.
    Returns None if the pattern doesn't match.
    """
    tensor = get_constant(ep, node)
    if tensor is not None:
        return tensor, node

    if node.target == exir_ops.edge.quantized_decomposed.quantize_per_tensor.default:
        source = node.args[0]
        assert isinstance(source, fx.Node)
        tensor = get_constant(ep, source)
        if tensor is not None:
            return tensor, node

    return None


def _resolve_constant_value(
    ep: ExportedProgram,
    op_node: fx.Node,
    qparams_prefix: str,
    constant_node: fx.Node,
) -> Optional[torch.Tensor]:
    constant_result = _trace_to_constant(ep, constant_node)
    if constant_result is None:
        return None
    constant = constant_result[0]
    if constant.is_floating_point():
        return constant

    qparams = get_qparams_from_node(op_node, qparams_prefix)
    if qparams is None:
        return constant
    scale = get_constant(ep, qparams.scale)
    zero_point = get_constant(ep, qparams.zero_point)
    if scale is None or zero_point is None:
        return None
    return QuantParamsStruct(
        scale=scale,
        zero_point=zero_point,
        dtype=qparams.dtype,
        quant_min=qparams.quant_min,
        quant_max=qparams.quant_max,
    ).dequantize(constant)


def _get_mul_data_and_scale(
    ep: ExportedProgram, mul_node: fx.Node
) -> Optional[tuple[fx.Node, torch.Tensor]]:
    inp_node = get_arg(mul_node, "inp", fx.Node)
    if mul_node.target == exir_ops.edge.fused_quant.mul.Scalar:
        return inp_node, torch.as_tensor(get_arg(mul_node, "other"))

    other_node = get_arg(mul_node, "other", fx.Node)
    scale_tensor = _resolve_constant_value(ep, mul_node, "other", other_node)
    if scale_tensor is not None:
        return inp_node, scale_tensor

    scale_tensor = _resolve_constant_value(ep, mul_node, "inp", inp_node)
    if scale_tensor is not None:
        return other_node, scale_tensor

    return None


def _iter_mul_data_and_scales(
    ep: ExportedProgram, graph: fx.Graph
) -> Iterator[tuple[fx.Node, fx.Node, torch.Tensor]]:
    mul_nodes = [
        *graph.find_nodes(
            op="call_function", target=exir_ops.edge.fused_quant.mul.default
        ),
        *graph.find_nodes(
            op="call_function", target=exir_ops.edge.fused_quant.mul.Scalar
        ),
    ]
    for mul_node in mul_nodes:
        data_and_scale = _get_mul_data_and_scale(ep, mul_node)
        if data_and_scale is not None:
            data_node, scale_tensor = data_and_scale
            if mul_node.meta["val"].shape == data_node.meta["val"].shape:
                yield mul_node, data_node, scale_tensor


class FuseMulIntoLinear(ExportedProgramPassBase):
    """Folds a fused_quant.mul with a constant operand into a preceding
    fused_quant linear or convolution by scaling its weight_scale and bias.

    Matches both fused_quant.mul tensor and Scalar overloads:
    - affine → fused_quant.mul(constant)
    - affine → [permute/view]* → fused_quant.mul(constant)

    Scalar constants are supported with per-tensor or per-channel weight qparams.
    Per-output-channel constants require private per-channel weight qparams and an
    output-channel axis that can be tracked through the passthroughs. Negative
    channels reflect the integer weights and zero points across the quantized range;
    zero channels use weight 0, scale 1, and zero point 0.
    """

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        graph = exported_program.graph_module.graph
        modified = False

        for mul_node, data_node, scale_tensor in _iter_mul_data_and_scales(
            exported_program, graph
        ):
            plan = _make_fold_plan(exported_program, data_node, scale_tensor)
            if plan is None:
                continue
            _apply_fold(exported_program, mul_node, data_node, plan)

            modified = True
            logger.info("Folded mul into affine: %s", plan.affine_node.name)

        if modified:
            # Doesn't hurt to run, and it also eliminates dead code and recompiles
            exported_program = constant_prop_pass(exported_program)

        return ExportedProgramPassResult(exported_program, modified)
