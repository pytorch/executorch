# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from executorch.backends.fused_quant.quantizer.qspecs import (
    arg1_is_scalar,
    make_qconfig,
)
from executorch.backends.fused_quant.quantizer.quantizer import (
    FusedQuantQuantizer,
    MaxPoolQuantizer,
    OpQuantizer,
    QuantizerBase,
)
from torchao.quantization.pt2e.quantizer import QuantizationConfig


# Why add/mul/sub are each registered twice -- once for tensor operands and once
# for a scalar operand:
#
# PyTorch does NOT give scalar arithmetic its own op overload. `tensor + 1e-5`
# (e.g. RMSNorm's `variance + eps`) traces to `aten.add.Tensor(tensor, 1e-5)` --
# the `.Tensor` overload carrying a Python scalar literal in arg 1, NOT a
# separate `aten.add.Scalar`. So the op type (`aten.add.Tensor`) alone does not
# tell us whether arg 1 is a tensor or a scalar; both land on the same op.
#
# Annotating a scalar arg as a quant edge breaks PT2E prepare, so the two operand
# shapes are handled separately:
#   - activation_names=("input", "other") quantizes both tensor operands; the
#     float-tensor guard makes it skip a scalar arg 1.
#   - activation_names=("input",) + node_filter=arg1_is_scalar quantizes only the
#     tensor input and leaves the scalar unquantized.
# Between them every add/mul/sub is covered. The scalar variant folds into
# `fused_quant.{add,mul,sub}.Scalar`, which pass the scalar through as a trailing
# arg.
def _get_binary_quantizers(config: QuantizationConfig) -> list[QuantizerBase]:
    quantizers: list[QuantizerBase] = []
    for aten_op, tensor_op, scalar_op in (
        (
            torch.ops.aten.add.Tensor,
            torch.ops.fused_quant.add.default,
            torch.ops.fused_quant.add.Scalar,
        ),
        (
            torch.ops.aten.mul.Tensor,
            torch.ops.fused_quant.mul.default,
            torch.ops.fused_quant.mul.Scalar,
        ),
        (
            torch.ops.aten.sub.Tensor,
            torch.ops.fused_quant.sub.default,
            torch.ops.fused_quant.sub.Scalar,
        ),
    ):
        quantizers.append(
            OpQuantizer(aten_op, tensor_op, config, activation_names=("input", "other"))
        )
        quantizers.append(
            OpQuantizer(
                aten_op,
                scalar_op,
                config,
                activation_names=("input",),
                node_filter=arg1_is_scalar,
            )
        )
    return quantizers


def get_core_quantizers(config: QuantizationConfig) -> list[QuantizerBase]:
    """Quantizers for the ops every backend is expected to handle.

    Everything is quantized with the single ``config``, so weights follow whatever
    granularity that config specifies. Use this for backends without per-channel
    support or without kernels for the wider op set.
    """
    return [
        *_get_binary_quantizers(config),
        OpQuantizer(
            torch.ops.aten.bmm.default,
            torch.ops.fused_quant.bmm.default,
            config,
            activation_names=("input", "mat2"),
        ),
        OpQuantizer(
            torch.ops.aten.relu.default, torch.ops.fused_quant.relu.default, config
        ),
        OpQuantizer(
            torch.ops.aten.linear.default,
            torch.ops.fused_quant.linear.default,
            config,
            activation_names=("input",),
            weight_names=("weight",),
            other_names=("bias",),
        ),
        OpQuantizer(
            torch.ops.aten.conv2d.default,
            torch.ops.fused_quant.conv2d.default,
            config,
            activation_names=("input",),
            weight_names=("weight",),
            other_names=("bias",),
        ),
    ]


def get_default_quantizers(
    default_config: QuantizationConfig,
    weighted_config: QuantizationConfig,
) -> list[QuantizerBase]:
    """Quantizers for every ATen op that has a ``fused_quant`` equivalent.

    Weighted ops (conv, linear) use ``weighted_config``; everything else uses
    ``default_config``. Both are ordinary ``QuantizationConfig`` values -- the
    split is by which ops they apply to, not by granularity, and nothing here
    requires either one to be per-tensor or per-channel.
    """
    weighted_ops = (
        (torch.ops.aten.conv1d.default, torch.ops.fused_quant.conv1d.default),
        (torch.ops.aten.conv2d.default, torch.ops.fused_quant.conv2d.default),
        (torch.ops.aten.conv3d.default, torch.ops.fused_quant.conv3d.default),
        (
            torch.ops.aten.convolution.default,
            torch.ops.fused_quant.convolution.default,
        ),
        (torch.ops.aten.linear.default, torch.ops.fused_quant.linear.default),
    )
    unary_ops = (
        (torch.ops.aten.hardswish.default, torch.ops.fused_quant.hardswish.default),
        (torch.ops.aten.sigmoid.default, torch.ops.fused_quant.sigmoid.default),
        (torch.ops.aten.tanh.default, torch.ops.fused_quant.tanh.default),
        (torch.ops.aten.hardtanh.default, torch.ops.fused_quant.hard_tanh.default),
        (torch.ops.aten.silu.default, torch.ops.fused_quant.silu.default),
        (
            torch.ops.aten.hardsigmoid.default,
            torch.ops.fused_quant.hardsigmoid.default,
        ),
        (torch.ops.aten.gelu.default, torch.ops.fused_quant.gelu.default),
        (
            torch.ops.aten.native_layer_norm.default,
            torch.ops.fused_quant.native_layer_norm.default,
        ),
        (torch.ops.aten.rms_norm.default, torch.ops.fused_quant.rms_norm.default),
        # Masked softmax: only the scores (input) and output are quantized; the
        # bool mask is left unquantized (threaded through, never quantized).
        (
            torch.ops.aten._masked_softmax.default,
            torch.ops.fused_quant._masked_softmax.default,
        ),
    )
    return [
        *_get_binary_quantizers(default_config),
        OpQuantizer(
            torch.ops.aten.bmm.default,
            torch.ops.fused_quant.bmm.default,
            default_config,
            activation_names=("input", "mat2"),
        ),
        OpQuantizer(
            torch.ops.aten.relu.default,
            torch.ops.fused_quant.relu.default,
            default_config,
        ),
        MaxPoolQuantizer(default_config),
        OpQuantizer(
            torch.ops.aten.avg_pool2d.default,
            torch.ops.fused_quant.avg_pool2d.default,
            default_config,
        ),
        *(
            OpQuantizer(
                aten_op,
                fused_op,
                weighted_config,
                activation_names=("input",),
                weight_names=("weight",),
                other_names=("bias",),
            )
            for aten_op, fused_op in weighted_ops
        ),
        *(
            OpQuantizer(aten_op, fused_op, default_config)
            for aten_op, fused_op in unary_ops
        ),
    ]


def make_fused_quant_quantizer(
    overrides: Optional[list[QuantizerBase]] = None,
    *,
    core_only: bool = False,
    weight_per_channel: bool = True,
    act_dtype: torch.dtype = torch.int8,
    weight_dtype: torch.dtype = torch.int8,
    act_symmetric: bool = False,
    weight_symmetric: bool = True,
    act_qmin: Optional[int] = None,
    act_qmax: Optional[int] = None,
    weight_qmin: Optional[int] = None,
    weight_qmax: Optional[int] = None,
) -> FusedQuantQuantizer:
    """Build a quantizer from override quantizers layered on top of the defaults.

    overrides are prepended to the defaults, so an overridden node is claimed
    by its override first (annotation is first-match-wins on key presence) and the
    default for that op skips it. An override is an ordinary OpQuantizer (or
    NoopQuantizer) carrying a node_filter that selects the nodes it owns.

    Op-type-level changes do not need an override: pass the desired dtype /
    symmetry / bounds knobs and the defaults handle them. Reach for an override
    only when a specific module or FX node needs a different config than others of
    the same op. Overrides are ordered; the first matching override claims the node.

    Args:
        core_only: Restrict the defaults to :func:`get_core_quantizers`.
        weight_per_channel: Quantize conv/linear weights per output channel.
        act_dtype / weight_dtype: Integer container dtypes for activations and
            weights (quantized independently). Default int8.
        act_symmetric / weight_symmetric: Use a symmetric qscheme (zero-point
            pinned to 0) for activations / weights.
        act_qmin / act_qmax / weight_qmin / weight_qmax: Explicit quant bounds.
            Activations and asymmetric weights default to the corresponding
            dtype's full ``torch.iinfo`` range. Signed symmetric weights default
            to a balanced range (e.g. -127/127 for int8).
    """
    knobs = {
        "act_dtype": act_dtype,
        "weight_dtype": weight_dtype,
        "act_symmetric": act_symmetric,
        "weight_symmetric": weight_symmetric,
        "act_qmin": act_qmin,
        "act_qmax": act_qmax,
        "weight_qmin": weight_qmin,
        "weight_qmax": weight_qmax,
    }
    config = make_qconfig(weight_per_channel=weight_per_channel, **knobs)
    defaults = (
        get_core_quantizers(config)
        if core_only
        else get_default_quantizers(
            make_qconfig(weight_per_channel=False, **knobs), config
        )
    )
    return FusedQuantQuantizer([*(overrides or []), *defaults])
