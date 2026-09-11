# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Provide shared operator groups and quantization helpers for Arm backends."""

import functools
from typing import Any, cast, NamedTuple

import torch
import torch.fx

from torch._ops import OpOverload
from torch.fx import Node
from torchao.quantization.pt2e import (
    FakeQuantize,
    FusedMovingAvgObsFakeQuantize,
    MovingAveragePerChannelMinMaxObserver,
    PartialWrapper,
)

from torchao.quantization.pt2e.quantizer import (
    FixedQParamsQuantizationSpec,
    QuantizationSpec,
    QuantizationSpecBase,
)


def _is_fused_moving_avg_obs_fake_quant_ctor(func: object) -> bool:
    """Return True when ``func`` is the fused fake-quant class or a subclass."""

    return isinstance(func, type) and issubclass(func, FusedMovingAvgObsFakeQuantize)


class _QParams(NamedTuple):
    scale: float
    zero_point: int
    quant_min: int | None = None
    quant_max: int | None = None


def _adjust_weight_qspec_for_conv_transpose(
    node: Node, weight_qspec: QuantizationSpec | None
) -> QuantizationSpec | None:
    """Adjust weight qspec axis/ctor for conv_transpose2d per-channel
    quantization.

    Use axis 1 for ungrouped ConvTranspose2d weights because the weight layout is
    (in_channels, out_channels / groups, kH, kW). Grouped transpose conv keeps axis 0.

    If the weight qspec contains a TorchAO QAT fake-quant/observer constructor
    (e.g. PartialWrapper(partial(...)) or a with_args-based constructor), the
    constructor is rebuilt with the corrected axis. For fused per-channel
    FakeQuantize, which only supports axis 0, the constructor is replaced with
    a non-fused FakeQuantize + MovingAveragePerChannelMinMaxObserver when the
    required axis is not 0.

    Return the qspec unchanged when weights are unset.

    """

    if (
        node.target != torch.ops.aten.conv_transpose2d.input
        or weight_qspec is None
        or weight_qspec.qscheme != torch.per_channel_symmetric
    ):
        return weight_qspec

    # For now skip axis adjustment for a8w4 per-channel configs (int4 weights).
    if weight_qspec.quant_min == -7 and weight_qspec.quant_max == 7:
        return weight_qspec

    groups = 1
    if len(node.args) > 6 and isinstance(node.args[6], int):
        groups = node.args[6]
    expected_axis = 0 if groups != 1 else 1

    observer_or_fake_quant_ctr = weight_qspec.observer_or_fake_quant_ctr
    observer_or_fake_quant_ctr_changed = False
    # QAT FakeQuantize uses PartialWrapper; rebuild its partial to update ch_axis
    # without breaking TorchAO introspection.
    if isinstance(observer_or_fake_quant_ctr, PartialWrapper):
        original_callable_args = dict(observer_or_fake_quant_ctr.callable_args)
        base_partial = observer_or_fake_quant_ctr.p
        if isinstance(base_partial, functools.partial):
            base_keywords = dict(base_partial.keywords or {})
            base_keywords["ch_axis"] = expected_axis
            if (
                _is_fused_moving_avg_obs_fake_quant_ctor(base_partial.func)
                and expected_axis != 0
            ):
                # Fused per-channel FakeQuant only supports axis 0; for other axes,
                # fall back to FakeQuantize with a per-channel observer.
                base_keywords["observer"] = MovingAveragePerChannelMinMaxObserver
                observer_or_fake_quant_ctr = PartialWrapper(
                    functools.partial(FakeQuantize, **base_keywords)
                )
            else:
                observer_or_fake_quant_ctr = PartialWrapper(
                    functools.partial(base_partial.func, **base_keywords)
                )
            observer_or_fake_quant_ctr.callable_args = original_callable_args
            observer_or_fake_quant_ctr_changed = True
    # Non-QAT observer/fake-quant ctrs can be updated via with_args.
    elif hasattr(observer_or_fake_quant_ctr, "with_args"):
        observer_or_fake_quant_ctr = observer_or_fake_quant_ctr.with_args(
            ch_axis=expected_axis
        )
        observer_or_fake_quant_ctr_changed = True

    if weight_qspec.ch_axis == expected_axis and not observer_or_fake_quant_ctr_changed:
        return weight_qspec

    return QuantizationSpec(
        dtype=weight_qspec.dtype,
        observer_or_fake_quant_ctr=observer_or_fake_quant_ctr,
        quant_min=weight_qspec.quant_min,
        quant_max=weight_qspec.quant_max,
        qscheme=weight_qspec.qscheme,
        ch_axis=expected_axis,
        is_dynamic=weight_qspec.is_dynamic,
    )


def _get_node_target(module: torch.nn.Module | torch.fx.GraphModule, target_str: str):
    """Get an attribute from a module by dotted path.

    Args:
        module (torch.nn.Module | torch.fx.GraphModule): Root module.
        target_str (str): Dotted attribute path, e.g., ``"sub.weight"``.

    Returns:
        Any: Resolved attribute on the module.

    """
    targets = target_str.split(".")
    for target in targets[:-1]:
        module = module.get_submodule(target)
    return getattr(module, targets[-1])


def _is_large_scalar(node: Node, gm: torch.fx.GraphModule):
    """Return True if input is a large scalar value.

    Large scalars are skipped because ``torch.histc`` supports values only up
    to a certain upper bound.

    """
    HISTC_UPPER_BOUND = 3.4028235e15
    if node.op == "get_attr" and isinstance(node.target, str):
        tensor = _get_node_target(gm, node.target)
        # torch.histc works until this upper bound
        return tensor.numel() == 1 and abs(tensor.item()) > HISTC_UPPER_BOUND
    if node.op == "call_function" and node.target in (
        torch.ops.aten.full.default,
        torch.ops.aten.full,
        torch.ops.aten.fill_.Scalar,
    ):
        fill_value = cast(float, node.args[1])
        return abs(fill_value) > HISTC_UPPER_BOUND
    return False


_conv_ops: set[OpOverload] = {
    torch.ops.aten.conv1d.default,
    torch.ops.aten.conv2d.default,
    torch.ops.aten.conv2d.padding,
    torch.ops.aten.conv_transpose2d.input,
    torch.ops.aten.conv3d.default,
    torch.ops.aten.conv3d.padding,
}

# For these ops, we use fixed qspecs, meaning that quantization params for
# these are statically defined. This is to prevent issues with out-of-range
# values when using dynamic quantization.
#
# Dict of operator to a dict of num_bits to qparams for that operator.
_fixed_input_qspec_ops: dict[Any, dict[int, _QParams]] = {
    # acos has a valid range of [-1, 1]
    torch.ops.aten.acos.default: {
        8: _QParams((1.0 - (-1.0)) / (1 << 8), 0),
        16: _QParams((1.0 - (-1.0)) / (1 << 16), 0),
    },
    # asin has a valid range of [-1, 1]
    torch.ops.aten.asin.default: {
        8: _QParams((1.0 - (-1.0)) / (1 << 8), 0),
        16: _QParams((1.0 - (-1.0)) / (1 << 16), 0),
    },
    # atanh has a valid range of (-1, 1) (excluding -1 and 1).
    torch.ops.aten.atanh.default: {
        8: _QParams((0.999 - (-0.999)) / (1 << 8), 0),
        16: _QParams((0.99999 - (-0.99999)) / (1 << 16), 0),
    },
    # grid_sampler image input/output use SNORM-compatible qparams. Input 1
    # follows the standard activation qspec, but the supported VGF lowering
    # modes are still only:
    # - float image / float grid / float output
    # - int8 image / int8 grid / int8 output
    # Mixed int8-image / float-grid shader lowering is not supported.
    torch.ops.aten.grid_sampler.default: {
        8: _QParams(1.0 / 127.0, 0, -127, 127),
    },
}


_fixed_output_qspec_ops: dict[Any, dict[int, _QParams]] = {
    torch.ops.aten.grid_sampler.default: {
        8: _QParams(1.0 / 127.0, 0, -127, 127),
    },
}


def _get_fixed_qparams_qspec(
    node_target: Any,
    qparams_table: dict[Any, dict[int, _QParams]],
    input_act_qspec: QuantizationSpecBase,
) -> FixedQParamsQuantizationSpec | None:
    if not isinstance(input_act_qspec, QuantizationSpec):
        raise ValueError("Fixed qparams require a QuantizationSpec input.")

    num_bits = torch.iinfo(input_act_qspec.dtype).bits
    qparams = qparams_table[node_target].get(num_bits)
    if qparams is None:
        return None

    return FixedQParamsQuantizationSpec(
        dtype=input_act_qspec.dtype,
        scale=qparams.scale,
        zero_point=qparams.zero_point,
        quant_min=(
            input_act_qspec.quant_min
            if qparams.quant_min is None
            else qparams.quant_min
        ),
        quant_max=(
            input_act_qspec.quant_max
            if qparams.quant_max is None
            else qparams.quant_max
        ),
        qscheme=input_act_qspec.qscheme,
        is_dynamic=input_act_qspec.is_dynamic,
    )


_one_to_one: set[OpOverload] = {
    torch.ops.aten.abs.default,
    torch.ops.aten.ceil.default,
    torch.ops.aten.erf.default,
    torch.ops.aten.erfinv.default,
    torch.ops.aten.exp.default,
    torch.ops.aten.expm1.default,
    torch.ops.aten.elu.default,
    torch.ops.aten.selu.default,
    torch.ops.aten.celu.default,
    torch.ops.aten.floor.default,
    torch.ops.aten.round.default,
    torch.ops.aten.log.default,
    torch.ops.aten.reciprocal.default,
    torch.ops.aten.rsqrt.default,
    torch.ops.aten.sigmoid.default,
    torch.ops.aten.cos.default,
    torch.ops.aten.sin.default,
    torch.ops.aten.tanh.default,
    torch.ops.aten.sum.dim_IntList,
    torch.ops.aten.sum.default,
    torch.ops.aten.hardsigmoid.default,
    torch.ops.aten.hardswish.default,
    torch.ops.aten.hardswish_.default,
    torch.ops.aten.leaky_relu.default,
    torch.ops.aten.leaky_relu_.default,
    torch.ops.aten.full_like.default,
    torch.ops.aten.zeros_like.default,
    torch.ops.aten.pow.Tensor_Scalar,
    torch.ops.aten.gelu.default,
    torch.ops.aten.silu.default,
    torch.ops.aten.silu_.default,
    torch.ops.aten.sinh.default,
    torch.ops.aten.atan.default,
    torch.ops.aten.log1p.default,
    torch.ops.aten.log10.default,
    torch.ops.aten.acosh.default,
    torch.ops.aten.sign.default,
    torch.ops.aten.asinh.default,
    torch.ops.aten.cosh.default,
    torch.ops.aten.cumsum.default,
    torch.ops.aten.remainder.Scalar,
    torch.ops.aten.tan.default,
}
