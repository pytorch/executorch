# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from torch.fx import Node
from torchao.quantization.pt2e import (
    HistogramObserver,
    MinMaxObserver,
    PerChannelMinMaxObserver,
)
from torchao.quantization.pt2e.quantizer import (
    DerivedQuantizationSpec,
    FixedQParamsQuantizationSpec,
    QuantizationSpec,
    SharedQuantizationSpec,
)

# ----------------- QUANTIZATION SPEC PRESETS -----------------
INT8_WEIGHT_PER_TENSOR_QSPEC = QuantizationSpec(
    dtype=torch.int8,
    observer_or_fake_quant_ctr=MinMaxObserver,
    qscheme=torch.per_tensor_symmetric,
)

INT8_WEIGHT_PER_CHANNEL_QSPEC = QuantizationSpec(
    dtype=torch.int8,
    observer_or_fake_quant_ctr=PerChannelMinMaxObserver,
    qscheme=torch.per_channel_symmetric,
    ch_axis=0,
)

# For transpose conv, output channels are at axis 1 (IOHW format vs OIHW for regular conv)
INT8_WEIGHT_PER_CHANNEL_TRANSPOSE_QSPEC = QuantizationSpec(
    dtype=torch.int8,
    observer_or_fake_quant_ctr=PerChannelMinMaxObserver,
    qscheme=torch.per_channel_symmetric,
    ch_axis=1,
)

INT8_ACTIVATION_PER_TENSOR_QSPEC = QuantizationSpec(
    dtype=torch.int8,
    observer_or_fake_quant_ctr=HistogramObserver,
    qscheme=torch.per_tensor_affine,
)

INT8_ACTIVATION_PER_CHANNEL_QSPEC = QuantizationSpec(
    dtype=torch.int8,
    observer_or_fake_quant_ctr=PerChannelMinMaxObserver,
    qscheme=torch.per_channel_affine,
    ch_axis=0,
)

# Constants shared by Cortex-M quantized operators.
CMSIS_SOFTMAX_SCALE: float = 1.0 / 256.0
CMSIS_SOFTMAX_ZERO_POINT: int = -128

SOFTMAX_OUTPUT_FIXED_QSPEC = FixedQParamsQuantizationSpec(
    dtype=torch.int8,
    scale=CMSIS_SOFTMAX_SCALE,
    zero_point=CMSIS_SOFTMAX_ZERO_POINT,
    quant_min=-128,
    quant_max=127,
    qscheme=torch.per_tensor_affine,
)

SOFTMAX_TARGETS = {
    torch.ops.aten._softmax.default,
    torch.ops.aten.softmax.int,
}

CONV_TRANSPOSE_TARGETS = {
    torch.ops.aten.conv_transpose2d.input,
}

POOL_SHARE_OUTPUT_TARGETS = {
    torch.ops.aten.avg_pool2d.default,
    torch.ops.aten.max_pool2d.default,
    torch.ops.aten.max_pool2d_with_indices.default,
}


class CortexMQuantizationConfig(QuantizationConfig):
    """Configures quantization, while enforcing cortex-m specific constraints."""

    def get_input_act_qspec(self, node: Node | None = None) -> QuantizationSpec | None:
        """
        Returns the configured input activation spec, no specific adjustments.
        """
        return super().get_input_act_qspec()

    def get_output_act_qspec(self, node: Node | None = None) -> QuantizationSpec | None:
        """
        Returns the configured output activation spec with the following cortex-m specific adjustments:
        - For softmax, returns a fixed quantization spec matching CMSIS-NN requirements.
        - For pooling ops, returns a SharedQuantizationSpec to indicate that the output should share the same quantization parameters as the input.
        """
        if node is not None and node.target in SOFTMAX_TARGETS:
            if self.output_activation is None:
                return None
            return SOFTMAX_OUTPUT_FIXED_QSPEC
        if node is not None and node.target in POOL_SHARE_OUTPUT_TARGETS:
            if len(node.args) == 0:
                return super().get_output_act_qspec()
            return SharedQuantizationSpec((node.args[0], node))
        return super().get_output_act_qspec()

    def get_weight_qspec(self, node: Node | None = None) -> QuantizationSpec | None:
        """
        Returns the configured weight quantization spec with the following cortex-m specific adjustments:
        - For conv transpose, returns the per-channel quantization spec with ch_axis=1 to match the IOHW weight format used by CMSIS-NN, instead of the default ch_axis=0
        """
        weight_qspec = super().get_weight_qspec()
        if (
            node is not None
            and node.target in CONV_TRANSPOSE_TARGETS
            and weight_qspec is not None
            and weight_qspec.dtype == torch.int8
        ):
            return INT8_WEIGHT_PER_CHANNEL_TRANSPOSE_QSPEC
        return weight_qspec

    def get_bias_qspec(self, node: Node) -> QuantizationSpec | None:
        """
        Returns the configured bias quantization spec, no specific adjustments.
        """
        if callable(self.bias):
            return self.bias(node)
        return super().get_bias_qspec(node)


def _derive_bias_qparams_fn(
    obs_or_fqs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(obs_or_fqs) != 2:
        raise ValueError(
            f"Expecting two obs/fqs, one for activation and one for weight, got: {len(obs_or_fqs)}"
        )
    act_obs_or_fq = obs_or_fqs[0]
    weight_obs_or_fq = obs_or_fqs[1]
    act_scale, _ = act_obs_or_fq.calculate_qparams()
    weight_scale, _ = weight_obs_or_fq.calculate_qparams()
    return act_scale * weight_scale, torch.full_like(
        weight_scale, fill_value=0, dtype=torch.int32
    )


def _get_int32_bias_qspec(node):
    return DerivedQuantizationSpec(
        derived_from=((node.args[0], node), (node.args[1], node)),  # type: ignore[list-item]
        derive_qparams_fn=_derive_bias_qparams_fn,
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.int32).min,
        quant_max=torch.iinfo(torch.int32).max - 1,
    )


def _get_int32_per_channel_bias_qspec(node):
    return DerivedQuantizationSpec(
        derived_from=((node.args[0], node), (node.args[1], node)),  # type: ignore[list-item]
        derive_qparams_fn=_derive_bias_qparams_fn,
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.int32).min,
        quant_max=torch.iinfo(torch.int32).max - 1,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
    )


# ----------------- QUANTIZATION CONFIG PRESETS -----------------
INT8_PER_TENSOR_CONFIG = CortexMQuantizationConfig(
    INT8_ACTIVATION_PER_TENSOR_QSPEC,
    INT8_ACTIVATION_PER_TENSOR_QSPEC,
    INT8_WEIGHT_PER_TENSOR_QSPEC,
    _get_int32_bias_qspec,
)


INT8_PER_CHANNEL_CONFIG = CortexMQuantizationConfig(
    INT8_ACTIVATION_PER_TENSOR_QSPEC,
    INT8_ACTIVATION_PER_TENSOR_QSPEC,
    INT8_WEIGHT_PER_CHANNEL_QSPEC,
    _get_int32_per_channel_bias_qspec,
)
