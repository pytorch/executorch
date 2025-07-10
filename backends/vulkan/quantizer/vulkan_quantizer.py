# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import functools
from typing import Callable, Optional

import torch
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer_utils import (
    _convert_scalars_to_attrs,
    OP_TO_ANNOTATOR,
    propagate_annotation,
)
from torch.fx import Node
from torchao.quantization.pt2e import (
    PerChannelMinMaxObserver,
    PlaceholderObserver,
)
from torchao.quantization.pt2e.quantizer import (
    QuantizationConfig,
    QuantizationSpec,
    Quantizer,
)


__all__ = [
    "VulkanQuantizer",
    "get_linear_weight_qcs_qspec",
    "get_linear_weight_only_qcs_xnn_qconfig",
]


def get_linear_weight_qcs_qspec(quant_bits: int) -> QuantizationSpec:
    """
    Return a QuantizationSpec to perform per-channel symmetric (i.e. "qcs") quantization
    of weight tensors of linear layers to the number of bits specified by quant_bits.
    """
    weight_observer = PerChannelMinMaxObserver
    assert quant_bits in {
        8,
        4,
    }, f"Unsupported weight quantization bits: {quant_bits}"

    quant_min = -(2 ** (quant_bits - 1))
    quant_max = 2 ** (quant_bits - 1) - 1
    qscheme = torch.per_channel_symmetric

    return QuantizationSpec(
        dtype=torch.int8,
        quant_min=quant_min,
        quant_max=quant_max,
        qscheme=qscheme,
        ch_axis=0,
        is_dynamic=False,
        observer_or_fake_quant_ctr=weight_observer,
    )


@functools.lru_cache
def get_linear_weight_only_qcs_xnn_qconfig(quant_bits: int) -> QuantizationConfig:
    """
    Return a XNNPACKQuantizer QuantizationConfig class instance that specifies
    quantizing the weight tensors of linear layers using per-channel symmetric (qcs)
    quantization to the number of bits specified by quant_bits.
    """
    weight_qspec = get_linear_weight_qcs_qspec(quant_bits)

    return QuantizationConfig(
        input_activation=None,
        output_activation=None,
        weight=weight_qspec,
        bias=None,
        is_qat=False,
    )


@functools.lru_cache
def get_dynamic_activation_qconfig(
    weight_bits: int = 4,
    act_qmin: int = -128,
    act_qmax: int = 127,
) -> QuantizationConfig:
    """
    Return a QuantizationConfig for dynamic activation quantization with 4-bit weights.
    This is compatible with Vulkan backend's quantized_decomposed operators.
    """
    # Dynamic activation quantization spec
    act_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=act_qmin,
        quant_max=act_qmax,
        qscheme=torch.per_tensor_affine,
        is_dynamic=True,
        observer_or_fake_quant_ctr=PlaceholderObserver,
    )

    # Weight quantization spec (per-channel symmetric)
    weight_qspec = get_linear_weight_qcs_qspec(weight_bits)

    return QuantizationConfig(
        input_activation=act_quantization_spec,
        output_activation=None,
        weight=weight_qspec,
        bias=None,
        is_qat=False,
    )


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
        # Support both static and dynamic quantization
        if self.global_config and self.global_config.input_activation and self.global_config.input_activation.is_dynamic:
            model = self._annotate_for_dynamic_quantization_config(model)
        else:
            model = self._annotate_for_static_quantization_config(model)
        propagate_annotation(model)
        return model

    def _annotate_all_patterns(
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
        self._annotate_all_patterns(
            model,
            self.global_config,
        )
        return model

    def _annotate_for_dynamic_quantization_config(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        self._annotate_all_patterns(
            model,
            self.global_config,
        )
        return model

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass
