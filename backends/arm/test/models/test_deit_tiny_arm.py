# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest
import timm  # type: ignore[import-untyped]

import torch

from executorch.backends.arm.common.pipeline_config import (  # type: ignore[attr-defined]
    ArmPassPipelineConfig,
    SDPASafeSoftmaxGuardPolicy,
)
from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

from timm.data import (  # type: ignore[import-untyped]
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from torchvision import transforms  # type: ignore[import-untyped]

normalize = transforms.Normalize(
    mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
)
model_inputs = (normalize(torch.rand((1, 3, 224, 224))),)
image_fp16 = torch.randn((1, 3, 224, 224), generator=torch.Generator().manual_seed(0))
model_inputs_fp16 = (normalize(image_fp16).to(torch.float16),)

input_t = Tuple[torch.Tensor]


@pytest.fixture(scope="module")
def deit_tiny():
    return timm.models.deit.deit_tiny_patch16_224(pretrained=True).eval()


@pytest.fixture(scope="module")
def get_deit_model(deit_tiny):
    models = {torch.float32: deit_tiny}

    def get_model(dtype):
        if dtype not in models:
            model = timm.models.deit.deit_tiny_patch16_224(pretrained=False)
            model.load_state_dict(deit_tiny.state_dict())
            models[dtype] = model.to(dtype).eval()
        return models[dtype]

    return get_model


fp_test_data = [
    pytest.param(
        torch.float32,
        model_inputs,
        {"atol": 1e-3, "rtol": 1e-3},
        id="fp32",
    ),
    pytest.param(
        torch.float16,
        model_inputs_fp16,
        {"atol": 8e-2, "rtol": 8e-2},
        marks=pytest.mark.slow,
        id="fp16",
    ),
]


@pytest.mark.parametrize("dtype,inputs,pipeline_kwargs", fp_test_data)
def test_deit_tiny_tosa_FP(dtype, inputs, pipeline_kwargs, get_deit_model):
    pipeline = TosaPipelineFP[input_t](
        get_deit_model(dtype),
        inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        **pipeline_kwargs,
    )
    pipeline.count_tosa_ops(
        {
            "EQUAL": 12,
            "LOGICAL_NOT": 24,
            "REDUCE_ANY": 12,
            "SELECT": 12,
        }
    )
    pipeline.run()


def test_deit_tiny_tosa_FP_remove_sdpa_safe_softmax_guard(deit_tiny):
    pipeline = TosaPipelineFP[input_t](
        deit_tiny,
        model_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.tester.compile_spec.set_pass_pipeline_config(
        ArmPassPipelineConfig(sdpa_safe_softmax_guard=SDPASafeSoftmaxGuardPolicy.REMOVE)
    )
    pipeline.count_tosa_ops(
        {
            "EQUAL": 0,
            "LOGICAL_NOT": 0,
            "REDUCE_ANY": 0,
            "SELECT": 0,
        }
    )
    pipeline.run()


def test_deit_tiny_tosa_INT(deit_tiny):
    pipeline = TosaPipelineINT[input_t](
        deit_tiny,
        model_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        atol=1.5,
        qtol=1,
        frobenius_threshold=None,
        cosine_threshold=None,
    )
    pipeline.run()


def test_deit_tiny_u55_INT(deit_tiny):
    pipeline = EthosU55PipelineINT[input_t](
        deit_tiny,
        model_inputs,
        aten_ops=[],
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
        atol=1.5,
        qtol=1,
    )
    # Multiple partitions
    pipeline.pop_stage("check_count.exir")
    # Don't run inference as model is too large for Corstone-300
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()


@common.XfailIfNoCorstone320
def test_deit_tiny_u85_INT(deit_tiny):
    pipeline = EthosU85PipelineINT[input_t](
        deit_tiny,
        model_inputs,
        aten_ops=[],
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
        atol=1.5,
        qtol=1,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_deit_tiny_vgf_quant(deit_tiny):
    pipeline = VgfPipeline[input_t](
        deit_tiny,
        model_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        atol=1.5,
        qtol=1,
        quantize=True,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@pytest.mark.parametrize("dtype,inputs,pipeline_kwargs", fp_test_data)
def test_deit_tiny_vgf_no_quant(dtype, inputs, pipeline_kwargs, get_deit_model):
    pipeline = VgfPipeline[input_t](
        get_deit_model(dtype),
        inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        quantize=False,
        **pipeline_kwargs,
    )
    pipeline.run()
