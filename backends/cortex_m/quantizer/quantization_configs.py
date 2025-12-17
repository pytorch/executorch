# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torchao.quantization.pt2e import (
    HistogramObserver,
    MinMaxObserver,
    PerChannelMinMaxObserver,
)
from torchao.quantization.pt2e.quantizer import (
    DerivedQuantizationSpec,
    QuantizationConfig,
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
