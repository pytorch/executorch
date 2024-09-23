# Copyright (c) 2024 MediaTek Inc.
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

import copy

from enum import IntEnum, unique

import torch

from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver
from torch.ao.quantization.quantizer import QuantizationSpec


@unique
class Precision(IntEnum):
    A16W16 = 0
    A16W8 = 1
    A16W4 = 2
    A8W8 = 3
    A8W4 = 4


class QuantizationConfig:

    def __init__(
        self, activation_spec: QuantizationSpec, weight_spec: QuantizationSpec
    ):
        self._activation_spec = activation_spec
        self._weight_spec = weight_spec

    @property
    def activation(self):
        return copy.deepcopy(self._activation_spec)

    @property
    def weight(self):
        return copy.deepcopy(self._weight_spec)


def get_quant_config(
    precision: Precision,
    is_per_channel: bool = False,
    is_qat: bool = False,
) -> QuantizationConfig:

    precision_mappings = {
        Precision.A16W16: get_a16w16_quant_config,
        Precision.A16W8: get_a16w8_quant_config,
        Precision.A16W4: get_a16w4_quant_config,
        Precision.A8W8: get_a8w8_quant_config,
        Precision.A8W4: get_a8w4_quant_config,
    }
    if precision not in precision_mappings:
        raise RuntimeError("Unrecognized precision setting.")

    qconfig_fn = precision_mappings[precision]
    return qconfig_fn(is_per_channel, is_qat)


def _get_activation_qspec(
    dtype,
    is_symmetric,
    is_qat,
    observer_cls=MinMaxObserver,
    quant_min=None,
    quant_max=None,
):
    if quant_max is None:
        quant_max = torch.iinfo(dtype).max
    if quant_min is None:
        # quant_min = torch.iinfo(dtype).min + 1 if is_symmetric else torch.iinfo(dtype).min
        quant_min = torch.iinfo(dtype).min

    qscheme = torch.per_tensor_symmetric if is_symmetric else torch.per_tensor_affine
    if is_qat:
        observer_or_fake_quant = FakeQuantize.with_args(observer=observer_cls, eps=1e-6)
    else:
        observer_or_fake_quant = observer_cls.with_args(eps=1e-6)

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
    if not is_per_channel:
        return _get_activation_qspec(
            dtype, is_symmetric, is_qat, observer_cls=MinMaxObserver
        )

    if quant_max is None:
        quant_max = torch.iinfo(dtype).max
    if quant_min is None:
        # quant_min = torch.iinfo(dtype).min + 1 if is_symmetric else torch.iinfo(dtype).min
        quant_min = torch.iinfo(dtype).min

    qscheme = torch.per_channel_symmetric if is_symmetric else torch.per_channel_affine
    if is_qat:
        observer_or_fake_quant = FakeQuantize.with_args(
            observer=PerChannelMinMaxObserver, eps=1e-6
        )
    else:
        observer_or_fake_quant = PerChannelMinMaxObserver.with_args(eps=1e-6)

    return QuantizationSpec(
        dtype=dtype,
        quant_min=quant_min,
        quant_max=quant_max,
        qscheme=qscheme,
        ch_axis=0,
        observer_or_fake_quant_ctr=observer_or_fake_quant,
    )


def get_a16w16_quant_config(is_per_channel, is_qat) -> QuantizationConfig:
    act_quantization_spec = _get_activation_qspec(torch.int16, True, is_qat)
    wgt_quantization_spec = _get_weight_qspec(torch.int16, True, is_per_channel, is_qat)
    quantization_config = QuantizationConfig(
        act_quantization_spec, wgt_quantization_spec
    )
    return quantization_config


def get_a16w8_quant_config(is_per_channel, is_qat) -> QuantizationConfig:
    act_quantization_spec = _get_activation_qspec(torch.int16, True, is_qat)
    wgt_quantization_spec = _get_weight_qspec(torch.int8, True, is_per_channel, is_qat)
    quantization_config = QuantizationConfig(
        act_quantization_spec, wgt_quantization_spec
    )
    return quantization_config


def get_a16w4_quant_config(is_per_channel, is_qat) -> QuantizationConfig:
    act_quantization_spec = _get_activation_qspec(torch.int16, True, is_qat)
    wgt_quantization_spec = _get_weight_qspec(
        torch.int8, False, is_per_channel, is_qat, quant_min=-8, quant_max=7
    )
    quantization_config = QuantizationConfig(
        act_quantization_spec, wgt_quantization_spec
    )
    return quantization_config


def get_a8w8_quant_config(is_per_channel, is_qat) -> QuantizationConfig:
    act_quantization_spec = _get_activation_qspec(torch.int8, False, is_qat)
    wgt_quantization_spec = _get_weight_qspec(torch.int8, False, is_per_channel, is_qat)
    quantization_config = QuantizationConfig(
        act_quantization_spec, wgt_quantization_spec
    )
    return quantization_config


def get_a8w4_quant_config(is_per_channel, is_qat) -> QuantizationConfig:
    act_quantization_spec = _get_activation_qspec(torch.int8, False, is_qat)
    wgt_quantization_spec = _get_weight_qspec(
        torch.int8, False, is_per_channel, is_qat, quant_min=-8, quant_max=7
    )
    quantization_config = QuantizationConfig(
        act_quantization_spec, wgt_quantization_spec
    )
    return quantization_config
