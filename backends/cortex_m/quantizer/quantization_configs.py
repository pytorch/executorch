# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from torchao.quantization.pt2e import (
    HistogramObserver,
    MinMaxObserver,
    PerChannelMinMaxObserver,
)
from torchao.quantization.pt2e.quantizer import (
    DerivedQuantizationSpec,
    FixedQParamsQuantizationSpec,
    QuantizationSpec,
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
        derived_from=[(node.args[0], node), (node.args[1], node)],  # type: ignore[list-item]
        derive_qparams_fn=_derive_bias_qparams_fn,
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.int32).min,
        quant_max=torch.iinfo(torch.int32).max - 1,
    )


def _get_int32_per_channel_bias_qspec(node):
    return DerivedQuantizationSpec(
        derived_from=[(node.args[0], node), (node.args[1], node)],  # type: ignore[list-item]
        derive_qparams_fn=_derive_bias_qparams_fn,
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.int32).min,
        quant_max=torch.iinfo(torch.int32).max - 1,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
    )


# ----------------- QUANTIZATION CONFIG PRESETS -----------------
INT8_PER_TENSOR_CONFIG = QuantizationConfig(
    INT8_ACTIVATION_PER_TENSOR_QSPEC,
    INT8_ACTIVATION_PER_TENSOR_QSPEC,
    INT8_WEIGHT_PER_TENSOR_QSPEC,
    _get_int32_bias_qspec,
)


INT8_PER_CHANNEL_CONFIG = QuantizationConfig(
    INT8_ACTIVATION_PER_TENSOR_QSPEC,
    INT8_ACTIVATION_PER_TENSOR_QSPEC,
    INT8_WEIGHT_PER_CHANNEL_QSPEC,
    _get_int32_per_channel_bias_qspec,
)


INT8_PER_CHANNEL_TRANSPOSE_CONFIG = QuantizationConfig(
    INT8_ACTIVATION_PER_TENSOR_QSPEC,
    INT8_ACTIVATION_PER_TENSOR_QSPEC,
    INT8_WEIGHT_PER_CHANNEL_TRANSPOSE_QSPEC,
    _get_int32_per_channel_bias_qspec,
)


SOFTMAX_PER_TENSOR_CONFIG = QuantizationConfig(
    INT8_ACTIVATION_PER_TENSOR_QSPEC,
    SOFTMAX_OUTPUT_FIXED_QSPEC,
    None,
    None,
)
