# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from dataclasses import dataclass

import torch

from torch.ao.quantization.quantizer import (
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

    def get_bias_qspec(self) -> QuantizationSpec | None:
        """Returns QuantizationSpec 'bias' after asserting that bias.dtype is torch.float."""
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
