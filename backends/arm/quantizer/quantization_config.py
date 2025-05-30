# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from dataclasses import dataclass

import torch
from torchao.quantization.pt2e import ObserverOrFakeQuantize

from torchao.quantization.pt2e.quantizer import (
    DerivedQuantizationSpec,
    FixedQParamsQuantizationSpec,
    QuantizationSpec,
)


@dataclass(eq=True, frozen=True)
class QuantizationConfig:
    input_activation: QuantizationSpec | None
    output_activation: QuantizationSpec | None
    weight: QuantizationSpec | None
    bias: QuantizationSpec | None

    def get_input_act_qspec(self) -> QuantizationSpec | None:
        """Returns QuantizationSpec 'input_activation' after asserting that input_activation.qscheme is valid."""
        if self.input_activation is None:
            return None
        assert self.input_activation.qscheme in [
            torch.per_tensor_affine,
            torch.per_tensor_symmetric,
        ], f"Unsupported quantization_spec {self.input_activation} for input_activation."
        return self.input_activation

    def get_output_act_qspec(self) -> QuantizationSpec | None:
        """Returns QuantizationSpec 'output_activation' after asserting that output_activation.qscheme is valid."""
        if self.output_activation is None:
            return None
        assert self.output_activation.qscheme in [
            torch.per_tensor_affine,
            torch.per_tensor_symmetric,
        ], f"Unsupported quantization_spec {self.output_activation} for output_activation."
        return self.output_activation

    def get_weight_qspec(self) -> QuantizationSpec | None:
        """Returns QuantizationSpec 'weight' after asserting that weight.qscheme is valid."""
        if self.weight is None:
            return None
        assert self.weight.qscheme in [
            torch.per_tensor_symmetric,
            torch.per_channel_symmetric,
        ], f"Unsupported quantization_spec {self.weight} for weight"
        return self.weight

    def get_bias_qspec(self, node: torch.fx.Node) -> QuantizationSpec | None:
        """Returns QuantizationSpec 'bias' after asserting that bias.dtype is torch.float."""

        def _derive_qparams_fn(
            obs_or_fqs: list[ObserverOrFakeQuantize],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            assert (
                len(obs_or_fqs) == 2
            ), "Expecting two obs/fqs, one for activation and one for weight, got: {}".format(
                len(obs_or_fqs)
            )
            act_obs_or_fq = obs_or_fqs[0]
            weight_obs_or_fq = obs_or_fqs[1]
            act_scale, act_zp = act_obs_or_fq.calculate_qparams()
            weight_scale, weight_zp = weight_obs_or_fq.calculate_qparams()
            return torch.tensor([act_scale * weight_scale]).to(
                torch.float32
            ), torch.tensor([0]).to(torch.int32)

        if node.target in [
            torch.ops.aten.conv1d.default,
            torch.ops.aten.conv2d.default,
            torch.ops.aten.linear.default,
            torch.ops.aten.conv2d.padding,
        ]:
            input_act = node.args[0]
            weight = node.args[1]
            quantization_spec = DerivedQuantizationSpec(
                derived_from=[(input_act, node), (weight, node)],  # type: ignore[list-item]
                derive_qparams_fn=_derive_qparams_fn,
                dtype=torch.int32,
                quant_min=torch.iinfo(torch.int32).min,
                quant_max=torch.iinfo(torch.int32).max - 1,
                qscheme=torch.per_tensor_symmetric,
            )
            return quantization_spec  # type: ignore[return-value]

        if self.bias is None:
            return None
        assert (
            self.bias.dtype == torch.float
        ), "Only float dtype for bias is supported for bias right now"
        return self.bias

    def get_fixed_qspec(
        self,
        scale: float,
        zp: int,
        dtype: torch.dtype = torch.int8,
        quant_min: int = -128,
        quant_max: int = 127,
    ) -> FixedQParamsQuantizationSpec:
        """Returns a new FixedQParamsQuantizationSpec with the given parameters."""
        return FixedQParamsQuantizationSpec(
            dtype=dtype,
            qscheme=torch.per_tensor_affine,
            scale=scale,
            zero_point=zp,
            quant_min=quant_min,
            quant_max=quant_max,
        )
