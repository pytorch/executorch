#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import Optional as _Optional

import torch as _torch

from attr import define as _define

from coremltools.optimize.torch.quantization.quantization_config import (
    ModuleLinearQuantizerConfig as _ModuleLinearQuantizerConfig,
    QuantizationScheme as _QuantizationScheme,
)

from torchao.quantization.pt2e.fake_quantize import FakeQuantize as _FakeQuantize

from torchao.quantization.pt2e.observer import (
    MinMaxObserver as _MinMaxObserver,
    MovingAverageMinMaxObserver as _MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver as _MovingAveragePerChannelMinMaxObserver,
    PerChannelMinMaxObserver as _PerChannelMinMaxObserver,
)
from torchao.quantization.pt2e.quantizer import (
    QuantizationSpec as _TorchQuantizationSpec,
)


def _get_observer(observer_type, is_per_channel: bool):
    _str_to_observer_map = {
        "moving_average_min_max": _MovingAverageMinMaxObserver,
        "min_max": _MinMaxObserver,
        "moving_average_min_max_per_channel": _MovingAveragePerChannelMinMaxObserver,
        "min_max_per_channel": _PerChannelMinMaxObserver,
    }
    observer_name = observer_type.value
    if is_per_channel:
        observer_name = f"{observer_name}_per_channel"
    if observer_name not in _str_to_observer_map:
        raise ValueError(f"Unsupported observer type: {observer_name}")
    return _str_to_observer_map[observer_name]


@_define
class AnnotationConfig:
    """
    Module/Operator level configuration class for :py:class:`CoreMLQuantizer`.

    For each module/operator, defines the dtype, quantization scheme and observer type
    for input(s), output and weights (if any).
    """

    input_activation: _Optional[_TorchQuantizationSpec] = None
    output_activation: _Optional[_TorchQuantizationSpec] = None
    weight: _Optional[_TorchQuantizationSpec] = None

    @staticmethod
    def _normalize_dtype(dtype: _torch.dtype) -> _torch.dtype:
        """
        PyTorch export quantizer only supports uint8 and int8 data types,
        so we map the quantized dtypes to the corresponding supported dtype.
        """
        dtype_map = {
            _torch.quint8: _torch.uint8,
            _torch.qint8: _torch.int8,
        }
        return dtype_map.get(dtype, dtype)

    @classmethod
    def from_quantization_config(
        cls,
        quantization_config: _Optional[_ModuleLinearQuantizerConfig],
    ) -> _Optional["AnnotationConfig"]:
        """
        Creates a :py:class:`AnnotationConfig` from ``ModuleLinearQuantizerConfig``
        """
        if (
            quantization_config is None
            or quantization_config.weight_dtype == _torch.float32
        ):
            return None

        # Activation QSpec
        if quantization_config.activation_dtype == _torch.float32:
            output_activation_qspec = None
        else:
            activation_qscheme = _QuantizationScheme.get_qscheme(
                quantization_config.quantization_scheme,
                is_per_channel=False,
            )
            activation_dtype = cls._normalize_dtype(
                quantization_config.activation_dtype
            )
            output_activation_qspec = _TorchQuantizationSpec(
                observer_or_fake_quant_ctr=_FakeQuantize.with_args(
                    observer=_get_observer(
                        quantization_config.activation_observer,
                        is_per_channel=False,
                    ),
                    dtype=activation_dtype,
                    qscheme=activation_qscheme,
                ),
                dtype=activation_dtype,
                qscheme=activation_qscheme,
            )

        # Weight QSpec
        weight_qscheme = _QuantizationScheme.get_qscheme(
            quantization_config.quantization_scheme,
            is_per_channel=quantization_config.weight_per_channel,
        )
        weight_dtype = cls._normalize_dtype(quantization_config.weight_dtype)
        weight_qspec = _TorchQuantizationSpec(
            observer_or_fake_quant_ctr=_FakeQuantize.with_args(
                observer=_get_observer(
                    quantization_config.weight_observer,
                    is_per_channel=quantization_config.weight_per_channel,
                ),
                dtype=weight_dtype,
                qscheme=weight_qscheme,
            ),
            dtype=weight_dtype,
            qscheme=weight_qscheme,
        )
        return AnnotationConfig(
            input_activation=output_activation_qspec,
            output_activation=output_activation_qspec,
            weight=weight_qspec,
        )
