# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from executorch.backends.qualcomm.quantizer.observers.per_block_param_observer import (
    PerBlockParamObserver,
)
from torch import Tensor
from torch.fx import Node
from torchao.quantization.pt2e import (
    FakeQuantize,
    FusedMovingAvgObsFakeQuantize,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    PerChannelMinMaxObserver,
)
from torchao.quantization.pt2e.quantizer import (
    DerivedQuantizationSpec,
    QuantizationSpec,
)


@dataclass(eq=True)
class QuantizationConfig:
    input_activation: Optional[QuantizationSpec]
    output_activation: Optional[QuantizationSpec]
    weight: Optional[QuantizationSpec]
    bias: Optional[QuantizationSpec | Callable]
    block_size: Optional[Tuple] = None


def _derived_bias_quant_spec(node: Node) -> DerivedQuantizationSpec:
    def _derive_bias_qparams_fn(
        obs_or_fqs: List,
    ) -> Tuple[Tensor, Tensor]:
        assert (
            len(obs_or_fqs) == 2
        ), f"Expecting two obs/fqs, one for activation and one for weight, got: {len(obs_or_fqs)}"
        act_obs_or_fq = obs_or_fqs[0]
        weight_obs_or_fq = obs_or_fqs[1]
        weight_scale, weight_zp = weight_obs_or_fq.calculate_qparams()
        act_scale, act_zp = act_obs_or_fq.calculate_qparams()
        (broadcast_act_scale, broadcast_weight_scale) = torch.broadcast_tensors(
            act_scale, weight_scale
        )
        derived_scale = (broadcast_act_scale * broadcast_weight_scale).to(torch.float32)
        # TransposeConv per channel axis=1, and the weight_shape[1] = out_channel / groups.
        # E.g., out_channel = 6, groups = 2, weight_shape[1] = 3, which means there are 3 pairs of scale/offset.
        # However, bias still has 6 values, meaning it requires repeat_interleave 2 times derived_scale in order to
        # generate 6 pairs of scale/offset to perform per channel quantization. For bias node, Conv OP builder will later
        # only pass 3 pairs of scale/offset to QNN.
        if (
            node.target
            in {
                torch.ops.aten.conv_transpose2d.input,
                torch.ops.aten.conv_transpose3d.input,
            }
            and len(node.args) > 6
            and node.args[6] != 1
        ):
            groups = node.args[6]
            derived_scale = derived_scale.repeat_interleave(groups)
        derived_zero = torch.zeros(derived_scale.size(), device=weight_zp.device).to(
            torch.int32
        )
        if isinstance(weight_obs_or_fq, PerBlockParamObserver):
            # keep maximum scale of each channel for bias
            derived_scale = (
                derived_scale.view(derived_scale.size(0), -1).amax(dim=-1)
                / weight_obs_or_fq.num_steps
            )
            derived_zero = derived_zero.view(derived_zero.size(0), -1).amax(dim=-1)
        return (derived_scale, derived_zero)

    input_act = node.args[0]
    assert isinstance(input_act, Node)
    weight = node.args[1]
    assert isinstance(weight, Node)
    return DerivedQuantizationSpec(
        derived_from=[(input_act, node), (weight, node)],
        derive_qparams_fn=_derive_bias_qparams_fn,
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.int32).min,
        quant_max=torch.iinfo(torch.int32).max,
        ch_axis=0,
        qscheme=torch.per_channel_symmetric,
    )


def get_8a8w_qnn_ptq_config(
    act_symmetric: bool = False, act_observer=MovingAverageMinMaxObserver
) -> QuantizationConfig:
    extra_args: Dict[str, Any] = {"eps": 2**-12}

    act_quantization_spec = QuantizationSpec(
        dtype=torch.uint8,
        qscheme=(
            torch.per_tensor_symmetric if act_symmetric else torch.per_tensor_affine
        ),
        ch_axis=0,
        observer_or_fake_quant_ctr=act_observer.with_args(**extra_args),
    )

    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=torch.iinfo(torch.int8).min + 1,
        quant_max=torch.iinfo(torch.int8).max,
        qscheme=torch.per_tensor_symmetric,
        ch_axis=0,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(**extra_args),
    )

    bias_quantization_spec = QuantizationSpec(
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.int32).min,
        quant_max=torch.iinfo(torch.int32).max,
        qscheme=torch.per_tensor_symmetric,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(**extra_args),
    )

    quantization_config = QuantizationConfig(
        input_activation=act_quantization_spec,
        output_activation=act_quantization_spec,
        weight=weight_quantization_spec,
        bias=bias_quantization_spec,
    )

    return quantization_config


def get_8a4w_qnn_ptq_config(
    act_symmetric: bool = True, act_observer=MovingAverageMinMaxObserver
) -> QuantizationConfig:
    extra_args: Dict[str, Any] = {"eps": 2**-12}

    if act_symmetric:
        # If zero_point is 128, htp can do optimizations.
        # If we keep quant_min and quant_max none, observer will default use 128 as zero_point.
        # If we provide uint8 quant_min/max, it will use 127 as zero_point, which is undesired.
        act_quantization_spec = QuantizationSpec(
            dtype=torch.uint8,
            qscheme=torch.per_tensor_symmetric,
            ch_axis=0,
            observer_or_fake_quant_ctr=act_observer.with_args(**extra_args),
        )
    else:
        # PyTorch will remove redundant observers based on attributes such as:
        # dtype, quant_min, quant_max, ch_axis, etc.
        # Providing values like quant_min and quant_max can help observers compare
        # and further reduce the number of observers.
        act_quantization_spec = QuantizationSpec(
            dtype=torch.uint8,
            quant_min=torch.iinfo(torch.uint8).min,
            quant_max=torch.iinfo(torch.uint8).max,
            qscheme=torch.per_tensor_affine,
            observer_or_fake_quant_ctr=act_observer.with_args(**extra_args),
        )

    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-7,
        quant_max=7,
        qscheme=torch.per_tensor_symmetric,
        ch_axis=0,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(**extra_args),
    )

    bias_quantization_spec = QuantizationSpec(
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.int32).min,
        quant_max=torch.iinfo(torch.int32).max,
        qscheme=torch.per_tensor_symmetric,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(**extra_args),
    )

    quantization_config = QuantizationConfig(
        input_activation=act_quantization_spec,
        output_activation=act_quantization_spec,
        weight=weight_quantization_spec,
        bias=bias_quantization_spec,
    )

    return quantization_config


# 4 bits quantization only supports specific ops.
def get_16a4w_qnn_ptq_config(
    act_observer=MovingAverageMinMaxObserver,
) -> QuantizationConfig:
    extra_args: Dict[str, Any] = {"eps": 2**-20}
    act_quantization_spec = QuantizationSpec(
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.uint16).min,
        quant_max=torch.iinfo(torch.uint16).max,
        qscheme=torch.per_tensor_affine,
        observer_or_fake_quant_ctr=act_observer.with_args(**extra_args),
    )

    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-7,
        quant_max=7,
        qscheme=torch.per_tensor_symmetric,
        ch_axis=0,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(**extra_args),
    )

    bias_quantization_spec = QuantizationSpec(
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.int32).min,
        quant_max=torch.iinfo(torch.int32).max,
        qscheme=torch.per_tensor_symmetric,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(**extra_args),
    )

    quantization_config = QuantizationConfig(
        input_activation=act_quantization_spec,
        output_activation=act_quantization_spec,
        weight=weight_quantization_spec,
        bias=bias_quantization_spec,
    )

    return quantization_config


def get_16a8w_qnn_ptq_config(
    act_observer=MovingAverageMinMaxObserver,
) -> QuantizationConfig:
    extra_args: Dict[str, Any] = {"eps": 2**-20}
    act_quantization_spec = QuantizationSpec(
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.uint16).min,
        quant_max=torch.iinfo(torch.uint16).max,
        qscheme=torch.per_tensor_affine,
        observer_or_fake_quant_ctr=act_observer.with_args(**extra_args),
    )

    weight_quantization_spec = QuantizationSpec(
        dtype=torch.uint8,
        qscheme=torch.per_tensor_symmetric,
        ch_axis=0,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(**extra_args),
    )

    bias_quantization_spec = QuantizationSpec(
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.int32).min,
        quant_max=torch.iinfo(torch.int32).max,
        qscheme=torch.per_tensor_symmetric,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(**extra_args),
    )

    quantization_config = QuantizationConfig(
        input_activation=act_quantization_spec,
        output_activation=act_quantization_spec,
        weight=weight_quantization_spec,
        bias=bias_quantization_spec,
    )

    return quantization_config


def get_16a8w_qnn_qat_config(
    act_observer=MovingAverageMinMaxObserver,
) -> QuantizationConfig:
    extra_args: Dict[str, Any] = {"eps": 2**-20}
    act_fake_quant_ctr = FusedMovingAvgObsFakeQuantize.with_args(
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.uint16).min,
        quant_max=torch.iinfo(torch.uint16).max,
        qscheme=torch.per_tensor_affine,
        observer=act_observer.with_args(**extra_args),
    )
    act_quantization_spec = QuantizationSpec(
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.uint16).min,
        quant_max=torch.iinfo(torch.uint16).max,
        qscheme=torch.per_tensor_affine,
        observer_or_fake_quant_ctr=act_fake_quant_ctr,
    )
    weight_fake_quant_ctr = FusedMovingAvgObsFakeQuantize.with_args(
        dtype=torch.int8,
        quant_min=torch.iinfo(torch.int8).min + 1,
        quant_max=torch.iinfo(torch.int8).max,
        qscheme=torch.per_tensor_symmetric,
        observer=MovingAverageMinMaxObserver,
    )
    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=torch.iinfo(torch.int8).min + 1,
        quant_max=torch.iinfo(torch.int8).max,
        qscheme=torch.per_tensor_symmetric,
        ch_axis=0,
        observer_or_fake_quant_ctr=weight_fake_quant_ctr,
    )
    bias_fake_quant_ctr = FakeQuantize.with_args(
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.int32).min,
        quant_max=torch.iinfo(torch.int32).max,
        qscheme=torch.per_tensor_symmetric,
        observer=MovingAverageMinMaxObserver,
    )
    bias_quantization_spec = QuantizationSpec(
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.int32).min,
        quant_max=torch.iinfo(torch.int32).max,
        qscheme=torch.per_tensor_symmetric,
        observer_or_fake_quant_ctr=bias_fake_quant_ctr,
    )
    quantization_config = QuantizationConfig(
        input_activation=act_quantization_spec,
        output_activation=act_quantization_spec,
        weight=weight_quantization_spec,
        bias=bias_quantization_spec,
    )

    return quantization_config


def get_16a16w_qnn_ptq_config(
    act_observer=MovingAverageMinMaxObserver,
) -> QuantizationConfig:
    extra_args: Dict[str, Any] = {"eps": 2**-20}
    act_quantization_spec = QuantizationSpec(
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.uint16).min,
        quant_max=torch.iinfo(torch.uint16).max,
        qscheme=torch.per_tensor_affine,
        observer_or_fake_quant_ctr=act_observer.with_args(**extra_args),
    )

    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int16,
        quant_min=torch.iinfo(torch.int16).min + 1,
        quant_max=torch.iinfo(torch.int16).max,
        qscheme=torch.per_tensor_symmetric,
        ch_axis=0,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(**extra_args),
    )

    # torch does not support uint16 quantization, use int32 to bypass
    bias_quantization_spec = QuantizationSpec(
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.int32).min,
        quant_max=torch.iinfo(torch.int32).max,
        qscheme=torch.per_tensor_symmetric,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(**extra_args),
    )

    quantization_config = QuantizationConfig(
        input_activation=act_quantization_spec,
        output_activation=act_quantization_spec,
        weight=weight_quantization_spec,
        bias=bias_quantization_spec,
    )

    return quantization_config


def get_ptq_per_channel_quant_config(
    act_dtype=torch.uint8,
    weight_dtype=torch.int8,
    act_observer=MovingAverageMinMaxObserver,
    act_symmetric: bool = False,
    ch_axis: int = 0,
) -> QuantizationConfig:
    extra_args: Dict[str, Any] = {"eps": 2**-12}

    supported_act_types = {
        torch.uint8,
        torch.uint16,
        torch.int8,
        torch.int16,
    }
    supported_weight_dtypes = {torch.int4, torch.int8, torch.int16}
    assert (
        act_dtype in supported_act_types
    ), f"act_dtype, {act_dtype} is not one of supported types, {supported_act_types}"

    assert (
        weight_dtype in supported_weight_dtypes
    ), f"weight_dtype, {weight_dtype} is not one of supported types, {supported_weight_dtypes}"

    # torch does not support uint16 quantization, use int32 to bypass
    if act_symmetric:
        # If zero_point is 128, htp can do optimizations.
        # If we keep quant_min and quant_max none, observer will default use 128 as zero_point.
        # If we provide uint8 quant_min/max, it will use 127 as zero_point, which is undesired.
        act_quantization_spec = QuantizationSpec(
            dtype=torch.int32 if act_dtype == torch.uint16 else act_dtype,
            qscheme=torch.per_tensor_symmetric,
            ch_axis=0,
            observer_or_fake_quant_ctr=act_observer.with_args(**extra_args),
        )
    else:
        # PyTorch will remove redundant observers based on attributes such as:
        # dtype, quant_min, quant_max, ch_axis, etc.
        # Providing values like quant_min and quant_max can help observers compare
        # and further reduce the number of observers.
        act_quantization_spec = QuantizationSpec(
            dtype=torch.int32 if act_dtype == torch.uint16 else act_dtype,
            quant_min=torch.iinfo(act_dtype).min,
            quant_max=torch.iinfo(act_dtype).max,
            qscheme=torch.per_tensor_affine,
            observer_or_fake_quant_ctr=act_observer.with_args(**extra_args),
        )

    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8 if weight_dtype == torch.int4 else weight_dtype,
        quant_min=(
            -7 if weight_dtype == torch.int4 else torch.iinfo(weight_dtype).min + 1
        ),
        quant_max=7 if weight_dtype == torch.int4 else torch.iinfo(weight_dtype).max,
        qscheme=torch.per_channel_symmetric,
        ch_axis=ch_axis,
        observer_or_fake_quant_ctr=PerChannelMinMaxObserver.with_args(**extra_args),
    )

    bias_quantization_spec = _derived_bias_quant_spec

    quantization_config = QuantizationConfig(
        input_activation=act_quantization_spec,
        output_activation=act_quantization_spec,
        weight=weight_quantization_spec,
        bias=bias_quantization_spec,
    )

    return quantization_config


def get_ptq_per_block_quant_config(
    act_dtype=torch.uint8,
    weight_dtype=torch.int8,
    act_observer=MovingAverageMinMaxObserver,
    act_symmetric: bool = False,
    ch_axis: int = 0,
) -> QuantizationConfig:
    extra_args: Dict[str, Any] = {"eps": 2**-12}
    quantization_config = get_ptq_per_channel_quant_config(
        act_dtype=act_dtype,
        weight_dtype=weight_dtype,
        act_observer=act_observer,
        act_symmetric=act_symmetric,
    )
    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8 if weight_dtype == torch.int4 else weight_dtype,
        quant_min=(
            -7 if weight_dtype == torch.int4 else torch.iinfo(weight_dtype).min + 1
        ),
        quant_max=7 if weight_dtype == torch.int4 else torch.iinfo(weight_dtype).max,
        qscheme=torch.per_channel_symmetric,
        ch_axis=ch_axis,
        observer_or_fake_quant_ctr=PerBlockParamObserver.with_args(**extra_args),
    )
    return QuantizationConfig(
        input_activation=quantization_config.input_activation,
        output_activation=quantization_config.output_activation,
        weight=weight_quantization_spec,
        bias=quantization_config.bias,
    )


# TODO merge qat and ptq to a function, and use a bool flag to control it
def get_8a8w_qnn_qat_config(
    act_symmetric: bool = False, act_observer=MovingAverageMinMaxObserver
) -> QuantizationConfig:
    act_fake_quant_ctr = FusedMovingAvgObsFakeQuantize.with_args(
        dtype=torch.uint8,
        qscheme=(
            torch.per_tensor_symmetric if act_symmetric else torch.per_tensor_affine
        ),
        observer=act_observer,
    )
    act_quantization_spec = QuantizationSpec(
        dtype=torch.uint8,
        qscheme=(
            torch.per_tensor_symmetric if act_symmetric else torch.per_tensor_affine
        ),
        ch_axis=0,
        observer_or_fake_quant_ctr=act_fake_quant_ctr,
    )

    weight_fake_quant_ctr = FusedMovingAvgObsFakeQuantize.with_args(
        dtype=torch.int8,
        quant_min=torch.iinfo(torch.int8).min + 1,
        quant_max=torch.iinfo(torch.int8).max,
        qscheme=torch.per_tensor_symmetric,
        observer=MovingAverageMinMaxObserver,
    )
    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=torch.iinfo(torch.int8).min + 1,
        quant_max=torch.iinfo(torch.int8).max,
        qscheme=torch.per_tensor_symmetric,
        ch_axis=0,
        observer_or_fake_quant_ctr=weight_fake_quant_ctr,
    )

    bias_fake_quant_ctr = FakeQuantize.with_args(
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.int32).min,
        quant_max=torch.iinfo(torch.int32).max,
        qscheme=torch.per_tensor_symmetric,
        observer=MovingAverageMinMaxObserver,
    )
    bias_quantization_spec = QuantizationSpec(
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.int32).min,
        quant_max=torch.iinfo(torch.int32).max,
        qscheme=torch.per_tensor_symmetric,
        observer_or_fake_quant_ctr=bias_fake_quant_ctr,
    )

    quantization_config = QuantizationConfig(
        input_activation=act_quantization_spec,
        output_activation=act_quantization_spec,
        weight=weight_quantization_spec,
        bias=bias_quantization_spec,
    )

    return quantization_config


def get_16a4w_qnn_qat_config(
    act_observer=MovingAverageMinMaxObserver,
) -> QuantizationConfig:
    act_fake_quant_ctr = FusedMovingAvgObsFakeQuantize.with_args(
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.uint16).min,
        quant_max=torch.iinfo(torch.uint16).max,
        qscheme=torch.per_tensor_affine,
        observer=act_observer,
    )
    act_quantization_spec = QuantizationSpec(
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.uint16).min,
        quant_max=torch.iinfo(torch.uint16).max,
        qscheme=torch.per_tensor_affine,
        observer_or_fake_quant_ctr=act_fake_quant_ctr,
    )

    weight_fake_quant_ctr = FusedMovingAvgObsFakeQuantize.with_args(
        dtype=torch.int8,
        quant_min=-7,
        quant_max=7,
        qscheme=torch.per_tensor_symmetric,
        ch_axis=0,
        observer=MovingAverageMinMaxObserver,
    )
    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-7,
        quant_max=7,
        qscheme=torch.per_tensor_symmetric,
        ch_axis=0,
        observer_or_fake_quant_ctr=weight_fake_quant_ctr,
    )

    bias_fake_quant_ctr = FakeQuantize.with_args(
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.int32).min,
        quant_max=torch.iinfo(torch.int32).max,
        qscheme=torch.per_tensor_symmetric,
        observer=MovingAverageMinMaxObserver,
    )
    bias_quantization_spec = QuantizationSpec(
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.int32).min,
        quant_max=torch.iinfo(torch.int32).max,
        qscheme=torch.per_tensor_symmetric,
        observer_or_fake_quant_ctr=bias_fake_quant_ctr,
    )

    quantization_config = QuantizationConfig(
        input_activation=act_quantization_spec,
        output_activation=act_quantization_spec,
        weight=weight_quantization_spec,
        bias=bias_quantization_spec,
    )

    return quantization_config


def get_qat_per_channel_quant_config(
    act_dtype=torch.uint8,
    weight_dtype=torch.int8,
    act_observer=MovingAverageMinMaxObserver,
    act_symmetric=False,
    ch_axis: int = 0,
) -> QuantizationConfig:
    supported_act_types = {
        torch.uint8,
        torch.uint16,
        torch.int8,
        torch.int16,
    }
    supported_weight_dtypes = {torch.int4, torch.int8, torch.int16}
    assert (
        act_dtype in supported_act_types
    ), f"act_dtype, {act_dtype} is not one of supported types, {supported_act_types}"

    assert (
        weight_dtype in supported_weight_dtypes
    ), f"weight_dtype, {weight_dtype} is not one of supported types, {supported_weight_dtypes}"

    # torch does not support uint16 quantization, use int32 to bypass
    if act_symmetric:
        # If zero_point is 128, htp can do optimizations.
        # If we keep quant_min and quant_max none, observer will default use 128 as zero_point.
        # If we provide uint8 quant_min/max, it will use 127 as zero_point, which is undesired.
        act_fake_quant_ctr = FusedMovingAvgObsFakeQuantize.with_args(
            dtype=torch.int32 if act_dtype == torch.uint16 else act_dtype,
            qscheme=torch.per_tensor_symmetric,
            observer=act_observer,
        )
        act_quantization_spec = QuantizationSpec(
            dtype=torch.int32 if act_dtype == torch.uint16 else act_dtype,
            qscheme=torch.per_tensor_symmetric,
            ch_axis=0,
            observer_or_fake_quant_ctr=act_fake_quant_ctr,
        )
    else:
        act_fake_quant_ctr = FusedMovingAvgObsFakeQuantize.with_args(
            dtype=torch.int32 if act_dtype == torch.uint16 else act_dtype,
            quant_min=torch.iinfo(act_dtype).min,
            quant_max=torch.iinfo(act_dtype).max,
            qscheme=torch.per_tensor_affine,
            observer=act_observer,
        )
        act_quantization_spec = QuantizationSpec(
            dtype=torch.int32 if act_dtype == torch.uint16 else act_dtype,
            quant_min=torch.iinfo(act_dtype).min,
            quant_max=torch.iinfo(act_dtype).max,
            qscheme=torch.per_tensor_affine,
            observer_or_fake_quant_ctr=act_fake_quant_ctr,
        )

    weight_fake_quant_ctr = FusedMovingAvgObsFakeQuantize.with_args(
        dtype=torch.int8 if weight_dtype == torch.int4 else weight_dtype,
        quant_min=(
            -7 if weight_dtype == torch.int4 else torch.iinfo(weight_dtype).min + 1
        ),
        quant_max=7 if weight_dtype == torch.int4 else torch.iinfo(weight_dtype).max,
        qscheme=torch.per_channel_symmetric,
        ch_axis=ch_axis,
        observer=MovingAveragePerChannelMinMaxObserver,
    )
    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8 if weight_dtype == torch.int4 else weight_dtype,
        quant_min=(
            -7 if weight_dtype == torch.int4 else torch.iinfo(weight_dtype).min + 1
        ),
        quant_max=7 if weight_dtype == torch.int4 else torch.iinfo(weight_dtype).max,
        qscheme=torch.per_channel_symmetric,
        ch_axis=ch_axis,
        observer_or_fake_quant_ctr=weight_fake_quant_ctr,
    )

    bias_quantization_spec = _derived_bias_quant_spec

    quantization_config = QuantizationConfig(
        input_activation=act_quantization_spec,
        output_activation=act_quantization_spec,
        weight=weight_quantization_spec,
        bias=bias_quantization_spec,
    )

    return quantization_config
