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
        # Validate that input_activation uses a supported qscheme
        if self.input_activation.qscheme not in [
            torch.per_tensor_affine,
            torch.per_tensor_symmetric,
        ]:
            raise ValueError(
                f"Unsupported quantization_spec {self.input_activation} for input_activation."
            )
        return self.input_activation

    def get_output_act_qspec(self) -> QuantizationSpec | None:
        """Returns QuantizationSpec 'output_activation' after asserting that output_activation.qscheme is valid."""
        if self.output_activation is None:
            return None
        # Validate that output_activation uses a supported qscheme
        if self.output_activation.qscheme not in [
            torch.per_tensor_affine,
            torch.per_tensor_symmetric,
        ]:
            raise ValueError(
                f"Unsupported quantization_spec {self.output_activation} for output_activation."
            )
        return self.output_activation

    def get_weight_qspec(self) -> QuantizationSpec | None:
        """Returns QuantizationSpec 'weight' after asserting that weight.qscheme is valid."""
        if self.weight is None:
            return None
        # Validate that weight uses a supported qscheme
        if self.weight.qscheme not in [
            torch.per_tensor_symmetric,
            torch.per_channel_symmetric,
        ]:
            raise ValueError(f"Unsupported quantization_spec {self.weight} for weight")
        return self.weight

    def get_bias_qspec(self, node: torch.fx.Node) -> QuantizationSpec | None:
        """Returns QuantizationSpec 'bias' after asserting that bias.dtype is torch.float."""

        def _derive_qparams_fn(
            obs_or_fqs: list[ObserverOrFakeQuantize],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # Validate expected number of observers/fake-quantizes
            if len(obs_or_fqs) != 2:
                raise ValueError(
                    f"Expecting two obs/fqs, one for activation and one for weight, got: {len(obs_or_fqs)}"
                )
            act_obs_or_fq = obs_or_fqs[0]
            weight_obs_or_fq = obs_or_fqs[1]
            act_scale, _ = act_obs_or_fq.calculate_qparams()
            weight_scale, _ = weight_obs_or_fq.calculate_qparams()
            return torch.tensor(act_scale * weight_scale).to(
                torch.float32
            ), torch.full_like(weight_scale, fill_value=0, dtype=torch.int32)

        if node.target in [
            torch.ops.aten.conv1d.default,
            torch.ops.aten.conv2d.default,
            torch.ops.aten.linear.default,
            torch.ops.aten.conv2d.padding,
        ]:
            input_act = node.args[0]
            weight = node.args[1]
            # If the weights are quantized per_tensor, do the same with bias
            qscheme = (
                torch.per_tensor_symmetric
                if self.weight is None
                else self.weight.qscheme
            )
            ch_axis = None
            if self.weight is not None:
                if qscheme == torch.per_channel_symmetric:
                    ch_axis = self.weight.ch_axis

            quantization_spec = DerivedQuantizationSpec(
                derived_from=[(input_act, node), (weight, node)],  # type: ignore[list-item]
                derive_qparams_fn=_derive_qparams_fn,
                dtype=torch.int32,
                quant_min=torch.iinfo(torch.int32).min,
                quant_max=torch.iinfo(torch.int32).max - 1,
                qscheme=qscheme,
                ch_axis=ch_axis,
            )
            return quantization_spec  # type: ignore[return-value]

        if self.bias is None:
            return None
        # Validate that bias dtype is floating-point
        if self.bias.dtype != torch.float:
            raise ValueError(
                "Only float dtype for bias is supported for bias right now"
            )
        return self.bias
