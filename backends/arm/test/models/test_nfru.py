# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Tuple

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from executorch.backends.arm.scripts.neural_graphics_test_data import (
    _NFRU_INPUT_CHANNELS,
    iter_nfru_test_calibration_samples,
    load_nfru_verification_inputs,
)

from executorch.backends.arm.test import common
from executorch.backends.arm.test.models.model_test_utils import (
    PTQ_AND_QAT_DATA,
    REAL_AND_RANDOM_DATA,
    skip_if_frozen_release,
)
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)
from huggingface_hub import hf_hub_download
from ng_model_gym.usecases.nfru.model.nfru_v1_nn import (  # type: ignore[import-not-found,import-untyped]
    NFRUAutoEncoder,
)

input_t = Tuple[torch.Tensor]  # Input x

pytestmark = skip_if_frozen_release("NFRU")


def nfru() -> NFRUAutoEncoder:
    """Get an instance of NFRU with FP32 weights loaded."""
    weights = hf_hub_download(  # nosec B615
        repo_id="Arm/neural-frame-rate-upscaling",
        filename="nfru_v1_fp32.pt",
        revision="main",
        cache_dir=os.environ.get("RUNNER_TEMP"),
    )
    checkpoint = torch.load(
        weights,
        map_location=torch.device("cpu"),
        weights_only=True,
    )["model_state_dict"]
    prefix = "network.auto_encoder."
    assert all(key.startswith(prefix) for key in checkpoint)

    model = NFRUAutoEncoder()
    model.load_state_dict(
        {key.removeprefix(prefix): value for key, value in checkpoint.items()},
        strict=True,
    )
    return model


def example_inputs():
    return load_nfru_verification_inputs()


def random_inputs(memory_format: torch.memory_format = torch.channels_last):
    return (
        torch.randn((1, _NFRU_INPUT_CHANNELS, 270, 480)).to(
            memory_format=memory_format
        ),
    )


input_test_data = REAL_AND_RANDOM_DATA
is_qat_test_data = PTQ_AND_QAT_DATA


def _set_nfru_calibration_samples(pipeline):
    return pipeline.set_quantization_calibration(iter_nfru_test_calibration_samples())


@common.parametrize("use_real_data", input_test_data)
def test_nfru_tosa_FP(use_real_data):
    pipeline = TosaPipelineFP[input_t](
        nfru().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("is_qat", is_qat_test_data)
@common.parametrize("use_real_data", input_test_data)
def test_nfru_tosa_INT(use_real_data, is_qat):
    pipeline_kwargs = (
        {
            "per_channel_quantization": False,
            "use_to_edge_transform_and_lower": True,
            "frobenius_threshold": None,
            "cosine_threshold": None,
        }
        if is_qat
        else {}
    )
    pipeline = TosaPipelineINT[input_t](
        nfru().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
        atol=0.2,
        qtol=2 if use_real_data else 1,
        is_qat=is_qat,
        **pipeline_kwargs,
    )
    if use_real_data:
        _set_nfru_calibration_samples(pipeline)
    pipeline.run()


@common.parametrize("use_real_data", input_test_data)
def test_nfru_tosa_INT_a16w8(use_real_data):
    pipeline = TosaPipelineINT[input_t](
        nfru().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
        tosa_extensions=["int16"],
        atol=0.1,
    )
    if use_real_data:
        _set_nfru_calibration_samples(pipeline)
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("use_real_data", input_test_data)
def test_nfru_vgf_no_quant(use_real_data):
    pipeline = VgfPipeline[input_t](
        nfru().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
        tosa_version="TOSA-1.0+FP",
        quantize=False,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("is_qat", is_qat_test_data)
@common.parametrize("use_real_data", input_test_data)
def test_nfru_vgf_quant(use_real_data, is_qat):
    pipeline_kwargs = (
        {
            "run_on_vulkan_runtime": True,
            "per_channel_quantization": False,
            "use_to_edge_transform_and_lower": True,
        }
        if is_qat
        else {}
    )
    pipeline = VgfPipeline[input_t](
        nfru().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
        tosa_version="TOSA-1.0+INT",
        symmetric_io_quantization=True,
        quantize=True,
        is_qat=is_qat,
        atol=0.2,
        qtol=2 if use_real_data else 1,
        **pipeline_kwargs,
    )
    if use_real_data:
        _set_nfru_calibration_samples(pipeline)
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("use_real_data", input_test_data)
def test_nfru_vgf_quant_a16w8(use_real_data):
    pipeline = VgfPipeline[input_t](
        nfru().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
        tosa_version="TOSA-1.0+INT",
        tosa_extensions=["int16"],
        symmetric_io_quantization=True,
        atol=0.2,
    )
    if use_real_data:
        _set_nfru_calibration_samples(pipeline)
    pipeline.run()
