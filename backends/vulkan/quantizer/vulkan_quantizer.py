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
from executorch.backends.vulkan.quantizer.vulkan_quantizer_utils import (
    _convert_scalars_to_attrs,
    bits_to_range,
    OP_TO_ANNOTATOR,
    propagate_annotation,
)
from torch.fx import Node
from torchao.quantization.pt2e import PerChannelMinMaxObserver, PlaceholderObserver
from torchao.quantization.pt2e.quantizer import (
    QuantizationConfig,
    QuantizationSpec,
    Quantizer,
)


__all__ = [
    "VulkanQuantizer",
    "get_symmetric_quantization_config",
]


@functools.lru_cache
def get_symmetric_quantization_config(
    is_dynamic: bool = False,
    weight_bits: int = 8,
    act_bits: int = 8,
    act_qmin: Optional[int] = None,
    act_qmax: Optional[int] = None,
    weight_qmin: Optional[int] = None,
    weight_qmax: Optional[int] = None,
) -> QuantizationConfig:
    """
    Return a QuantizationConfig for Vulkan quantizer.

    Args:
        is_dynamic: If False, weight-only quantization. If True, dynamic quantization (activation + weight)
        weight_bits: Number of bits for weight quantization (4 or 8)
        act_bits: Number of bits for activation quantization (8)
        act_qmin: Minimum quantization value for activations (auto-calculated if None)
        act_qmax: Maximum quantization value for activations (auto-calculated if None)
        weight_qmin: Minimum quantization value for weights (auto-calculated if None)
        weight_qmax: Maximum quantization value for weights (auto-calculated if None)
    """
    assert weight_bits in {
        8,
        4,
    }, f"Unsupported weight quantization bits: {weight_bits}"

    assert act_bits in {
        8,
    }, f"Unsupported activation quantization bits: {act_bits}"

    # Auto-calculate weight ranges if not provided
    if weight_qmin is None or weight_qmax is None:
        weight_range = bits_to_range(weight_bits)
        weight_qmin = weight_qmin if weight_qmin is not None else weight_range[0]
        weight_qmax = weight_qmax if weight_qmax is not None else weight_range[1]

    # Weight quantization: per-channel symmetric for Vulkan
    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=weight_qmin,
        quant_max=weight_qmax,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
        is_dynamic=False,
        observer_or_fake_quant_ctr=PerChannelMinMaxObserver,
    )

    # Configure activation quantization based on is_dynamic
    if not is_dynamic:
        # Weight-only quantization: no activation quantization
        act_quantization_spec = None
        output_activation_spec = None
    else:
        # Dynamic quantization: per-token input quantization, no output quantization
        # Auto-calculate activation ranges if not provided
        if act_qmin is None or act_qmax is None:
            act_range = bits_to_range(act_bits)
            act_qmin = act_qmin if act_qmin is not None else act_range[0]
            act_qmax = act_qmax if act_qmax is not None else act_range[1]

        act_observer_or_fake_quant_ctr = PlaceholderObserver
        act_quantization_spec = QuantizationSpec(
            dtype=torch.int8,
            quant_min=act_qmin,
            quant_max=act_qmax,
            qscheme=torch.per_tensor_affine,
            is_dynamic=True,
            observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr,
        )
        output_activation_spec = None

    return QuantizationConfig(
        input_activation=act_quantization_spec,
        output_activation=output_activation_spec,
        weight=weight_quantization_spec,
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
        model = self._annotate_for_quantization_config(model)
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

    def _annotate_for_quantization_config(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        self._annotate_all_patterns(
            model,
            self.global_config,
        )
        return model

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass
