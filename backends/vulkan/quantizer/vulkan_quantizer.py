# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, Optional

import torch
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.ao.quantization.quantizer import QuantizationSpec, Quantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
    _convert_scalars_to_attrs,
    OP_TO_ANNOTATOR,
    propagate_annotation,
    QuantizationConfig,
)
from torch.fx import Node


__all__ = [
    "VulkanQuantizer",
    "get_weight_quantization_config",
]


@functools.lru_cache
def get_weight_quantization_config(
    is_per_channel: bool = True,
    weight_qmin: int = -128,
    weight_qmax: int = 127,
) -> QuantizationConfig:

    weight_qscheme = (
        torch.per_channel_symmetric if is_per_channel else torch.per_tensor_symmetric
    )
    weight_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = (
        PerChannelMinMaxObserver if is_per_channel else MinMaxObserver
    )
    extra_args: Dict[str, Any] = {"eps": 2**-12}

    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=weight_qmin,
        quant_max=weight_qmax,
        qscheme=weight_qscheme,
        ch_axis=0,
        is_dynamic=False,
        observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr.with_args(
            **extra_args
        ),
    )

    quantization_config = QuantizationConfig(
        input_activation=None,
        output_activation=None,
        weight=weight_quantization_spec,
        bias=None,
        is_qat=False,
    )
    return quantization_config


_SUPPORTED_OPS = [
    "linear",
]


class VulkanQuantizer(Quantizer):

    def __init__(self) -> None:
        super().__init__()
        self.global_config: Optional[QuantizationConfig] = None

    def set_global(self, quantization_config: QuantizationConfig) -> VulkanQuantizer:
        self.global_config = quantization_config
        return self

    def transform_for_annotation(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        """Transforms scalar values to tensor attributes"""
        return _convert_scalars_to_attrs(model)

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        # currently only support static quant on Vulkan
        model = self._annotate_for_static_quantization_config(model)
        propagate_annotation(model)
        return model

    def _annotate_all_static_patterns(
        self,
        model: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[Callable[[Node], bool]] = None,
    ) -> torch.fx.GraphModule:
        if quantization_config is None:
            return model

        for op in _SUPPORTED_OPS:
            OP_TO_ANNOTATOR[op](model, quantization_config, filter_fn)
        return model

    def _annotate_for_static_quantization_config(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        self._annotate_all_static_patterns(
            model,
            self.global_config,
        )
        return model

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass
