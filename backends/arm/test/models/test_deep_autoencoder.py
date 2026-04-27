# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Ethos-U FVP tests for the MLPerf Tiny anomaly detection Deep AutoEncoder."""

from typing import Tuple

import pytest
import torch
import torch.nn as nn
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
)

from executorch.examples.models.mlperf_tiny import DeepAutoEncoderModel
from torch.nn.utils.fusion import fuse_linear_bn_eval


def _fuse_linear_bn(mod: nn.Module) -> nn.Module:
    """Fuse Linear + BatchNorm1d pairs in the model.

    The TOSA quantizer does not annotate linear+batch_norm patterns, so we fold
    the BatchNorm1d into the preceding Linear before export.
    TODO: Remove once the quantizer supports linear+bn.

    """
    if not isinstance(mod, nn.Sequential):
        for name, child in mod.named_children():
            setattr(mod, name, _fuse_linear_bn(child))
        return mod
    new_layers = []
    layers = list(mod)
    i = 0
    while i < len(layers):
        if (
            isinstance(layers[i], nn.Linear)
            and i + 1 < len(layers)
            and isinstance(layers[i + 1], nn.BatchNorm1d)
        ):
            new_layers.append(fuse_linear_bn_eval(layers[i], layers[i + 1]))  # type: ignore[type-var, arg-type]
            i += 2
        else:
            new_layers.append(_fuse_linear_bn(layers[i]))
            i += 1
    return nn.Sequential(*new_layers)


_wrapper = DeepAutoEncoderModel()
model = _fuse_linear_bn(_wrapper.get_eager_model())
model_inputs = _wrapper.get_example_inputs()
input_t = Tuple[torch.Tensor]

quant_test_data = {
    "per_channel_quantization=true": True,
    "per_channel_quantization=false": False,
}


def test_deep_autoencoder_tosa_FP():
    pipeline = TosaPipelineFP[input_t](
        model,
        model_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@common.parametrize("per_channel_quantization", quant_test_data)
def test_deep_autoencoder_tosa_INT(per_channel_quantization):
    pipeline = TosaPipelineINT[input_t](
        model,
        model_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        per_channel_quantization=per_channel_quantization,
        atol=0.25,
        qtol=1,
        frobenius_threshold=None,
        cosine_threshold=None,
    )
    pipeline.run()


@pytest.mark.slow
@common.XfailIfNoCorstone300
@common.parametrize("per_channel_quantization", quant_test_data)
def test_deep_autoencoder_u55_INT(per_channel_quantization):
    pipeline = EthosU55PipelineINT[input_t](
        model,
        model_inputs,
        aten_ops=[],
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
        per_channel_quantization=per_channel_quantization,
        atol=0.25,
        qtol=1,
    )
    pipeline.run()


@pytest.mark.slow
@common.XfailIfNoCorstone320
@common.parametrize("per_channel_quantization", quant_test_data)
def test_deep_autoencoder_u85_INT(per_channel_quantization):
    pipeline = EthosU85PipelineINT[input_t](
        model,
        model_inputs,
        aten_ops=[],
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
        per_channel_quantization=per_channel_quantization,
        atol=0.25,
        qtol=1,
    )
    pipeline.run()
