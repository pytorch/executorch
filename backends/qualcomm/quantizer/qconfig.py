from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.ao.quantization.fake_quantize import (
    FakeQuantize,
    FusedMovingAvgObsFakeQuantize,
)
from torch.ao.quantization.observer import (
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    PerChannelMinMaxObserver,
)
from torch.ao.quantization.quantizer import DerivedQuantizationSpec, QuantizationSpec
from torch.fx import Node


@dataclass(eq=True, frozen=True)
class QuantizationConfig:
    input_activation: Optional[QuantizationSpec]
    output_activation: Optional[QuantizationSpec]
    weight: Optional[QuantizationSpec]
    bias: Optional[QuantizationSpec | Callable]


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
        derived_zero = torch.zeros(derived_scale.size()).to(torch.int32)
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
) -> QuantizationConfig:
    extra_args: Dict[str, Any] = {"eps": 2**-12}

    supported_act_types = {
        torch.uint8,
        torch.uint16,
        torch.int8,
        torch.int16,
    }
    # TODO accept "int4" temporally. Remove "int4" when torch support torch.int4 dtype
    supported_weight_dtypes = {"int4", torch.int8, torch.int16}
    assert (
        act_dtype in supported_act_types
    ), f"act_dtype, {act_dtype} is not one of supported types, {supported_act_types}"

    assert (
        weight_dtype in supported_weight_dtypes
    ), f"weight_dtype, {weight_dtype} is not one of supported types, {supported_weight_dtypes}"

    # torch do not support uint16 quantization, use int32 to bypass
    if act_symmetric:
        # If zero_point is 128, htp can do optimizations.
        # If we keep quant_min and quant_max none, observer will default use 128 as zero_point.
        # If we provide uint8 quant_min/max, it will use 127 as zero_point, which is undesired.
        act_quantization_spec = QuantizationSpec(
            dtype=torch.int32 if act_dtype == torch.uint16 else act_dtype,
            qscheme=torch.per_tensor_symmetric,
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
        dtype=torch.int8 if weight_dtype == "int4" else weight_dtype,
        quant_min=-7 if weight_dtype == "int4" else torch.iinfo(weight_dtype).min + 1,
        quant_max=7 if weight_dtype == "int4" else torch.iinfo(weight_dtype).max,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
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


# TODO merge qat and ptq to a fucntion, and use a bool flag to control it
def get_8a8w_qnn_qat_config(
    act_symmetric: bool = False, act_observer=MovingAverageMinMaxObserver
) -> QuantizationConfig:
    act_fake_quant_ctr = FakeQuantize.with_args(
        dtype=torch.uint8,
        qscheme=(
            torch.per_tensor_symmetric if act_symmetric else torch.per_tensor_affine
        ),
        reduce_range=True,
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
        reduce_range=True,
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
        reduce_range=True,
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
    act_fake_quant_ctr = FakeQuantize.with_args(
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.uint16).min,
        quant_max=torch.iinfo(torch.uint16).max,
        qscheme=torch.per_tensor_affine,
        reduce_range=True,
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
        reduce_range=True,
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
        reduce_range=True,
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
) -> QuantizationConfig:
    supported_act_types = {
        torch.uint8,
        torch.uint16,
        torch.int8,
        torch.int16,
    }
    # TODO accept "int4" temporally. Remove "int4" when torch support torch.int4 dtype
    supported_weight_dtypes = {"int4", torch.int8, torch.int16}
    assert (
        act_dtype in supported_act_types
    ), f"act_dtype, {act_dtype} is not one of supported types, {supported_act_types}"

    assert (
        weight_dtype in supported_weight_dtypes
    ), f"weight_dtype, {weight_dtype} is not one of supported types, {supported_weight_dtypes}"

    # torch do not support uint16 quantization, use int32 to bypass
    act_fake_quant_ctr = FakeQuantize.with_args(
        dtype=torch.int32 if act_dtype == torch.uint16 else act_dtype,
        quant_min=torch.iinfo(act_dtype).min,
        quant_max=torch.iinfo(act_dtype).max,
        qscheme=torch.per_tensor_affine,
        reduce_range=True,
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
        dtype=torch.int8 if weight_dtype == "int4" else weight_dtype,
        quant_min=-7 if weight_dtype == "int4" else torch.iinfo(weight_dtype).min + 1,
        quant_max=7 if weight_dtype == "int4" else torch.iinfo(weight_dtype).max,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
        observer=MovingAveragePerChannelMinMaxObserver,
    )
    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8 if weight_dtype == "int4" else weight_dtype,
        quant_min=-7 if weight_dtype == "int4" else torch.iinfo(weight_dtype).min + 1,
        quant_max=7 if weight_dtype == "int4" else torch.iinfo(weight_dtype).max,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
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
