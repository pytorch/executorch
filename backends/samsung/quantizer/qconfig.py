# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import IntEnum, unique
from typing import Callable, Optional

import torch
from torchao.quantization.pt2e import (
    FusedMovingAvgObsFakeQuantize,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    PerChannelMinMaxObserver,
)
from torchao.quantization.pt2e.quantizer import QuantizationSpec


@unique
class Precision(IntEnum):
    A8W8 = 3


@dataclass(eq=True, frozen=True)
class QuantizationConfig:
    input_activation: Optional[QuantizationSpec]
    output_activation: Optional[QuantizationSpec]
    weight: Optional[QuantizationSpec]
    bias: Optional[QuantizationSpec | Callable]


def get_quant_config(
    precision: Precision,
    is_per_channel: bool = False,
    is_qat: bool = False,
) -> QuantizationConfig:

    precision_mappings = {
        Precision.A8W8: get_a8w8_enn_quant_config,
    }
    if precision not in precision_mappings:
        raise RuntimeError("Unrecognized precision setting.")

    is_weight_symm = is_per_channel

    qconfig_fn = precision_mappings[precision]
    return qconfig_fn(is_per_channel, is_qat, wei_symmetric=is_weight_symm)


def _get_activation_qspec(
    dtype,
    is_symmetric,
    is_qat,
    observer_cls=MinMaxObserver,
    quant_min=None,
    quant_max=None,
):
    eps_value = 2**-12
    if quant_max is None:
        quant_max = torch.iinfo(dtype).max
    if quant_min is None:
        quant_min = torch.iinfo(dtype).min

    qscheme = torch.per_tensor_symmetric if is_symmetric else torch.per_tensor_affine
    if is_qat:
        observer_or_fake_quant = FusedMovingAvgObsFakeQuantize.with_args(eps=eps_value)
    else:
        observer_or_fake_quant = observer_cls.with_args(eps=eps_value)

    return QuantizationSpec(
        dtype=dtype,
        quant_min=quant_min,
        quant_max=quant_max,
        qscheme=qscheme,
        observer_or_fake_quant_ctr=observer_or_fake_quant,
    )


def _get_weight_qspec(
    dtype, is_symmetric, is_per_channel, is_qat, quant_min=None, quant_max=None
):
    assert is_symmetric or not is_per_channel, "Not support asymm+perchannel mode"

    eps_value = 2**-12

    if quant_max is None:
        quant_max = torch.iinfo(dtype).max
    if quant_min is None:
        quant_min = torch.iinfo(dtype).min

    if not is_per_channel:
        qscheme = (
            torch.per_tensor_symmetric if is_symmetric else torch.per_tensor_affine
        )
        observer_cls = MinMaxObserver
    else:
        qscheme = (
            torch.per_channel_symmetric if is_symmetric else torch.per_channel_affine
        )
        observer_cls = PerChannelMinMaxObserver

    if is_qat:
        observer_cls = FusedMovingAvgObsFakeQuantize
        if not is_per_channel:
            weight_qat_observer = MovingAverageMinMaxObserver
        else:
            weight_qat_observer = MovingAveragePerChannelMinMaxObserver
        observer_or_fake_quant = observer_cls.with_args(
            eps=eps_value,
            observer=weight_qat_observer,
        )
    else:
        observer_or_fake_quant = observer_cls.with_args(eps=eps_value)

    return QuantizationSpec(
        dtype=dtype,
        quant_min=quant_min,
        quant_max=quant_max,
        qscheme=qscheme,
        ch_axis=0,
        observer_or_fake_quant_ctr=observer_or_fake_quant,
    )


def get_a8w8_enn_quant_config(
    is_per_channel=True, is_qat=False, act_symmetric=False, wei_symmetric=False
) -> QuantizationConfig:
    act_quantization_spec = _get_activation_qspec(torch.int8, act_symmetric, is_qat)
    wgt_quantization_spec = _get_weight_qspec(
        torch.int8, wei_symmetric, is_per_channel, is_qat
    )
    bias_quantization_spec = None
    quantization_config = QuantizationConfig(
        input_activation=act_quantization_spec,
        output_activation=act_quantization_spec,
        weight=wgt_quantization_spec,
        bias=bias_quantization_spec,
    )
    return quantization_config
