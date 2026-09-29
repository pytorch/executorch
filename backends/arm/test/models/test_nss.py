# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest
import torch
from executorch.backends.arm.scripts.neural_graphics_test_data import (
    _NSS_INPUT_CHANNELS,
    iter_nss_test_calibration_samples,
    load_nss_verification_inputs,
)

from executorch.backends.arm.test import common
from executorch.backends.arm.test.models.model_test_utils import (
    PTQ_AND_QAT_DATA,
    REAL_AND_RANDOM_DATA,
    skip_if_frozen_release,
)
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

from huggingface_hub import hf_hub_download

from ng_model_gym.usecases.nss.model.model_blocks_v1 import (  # type: ignore[import-not-found,import-untyped]
    AutoEncoderV1,
)
from torch.export import Dim

input_t = Tuple[torch.Tensor]  # Input x

pytestmark = skip_if_frozen_release("NSS")

_NSS_HEIGHT = 8 * Dim("_nss_height", min=16, max=68)
_NSS_WIDTH = 8 * Dim("_nss_width", min=16, max=120)
_NSS_QUANTIZATION_DYNAMIC_SHAPES = ({2: _NSS_HEIGHT, 3: _NSS_WIDTH},)


class NSS(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auto_encoder = AutoEncoderV1()


def nss() -> AutoEncoderV1:
    """Get an instance of NSS with weights loaded."""

    weights = hf_hub_download(  # nosec B615
        repo_id="Arm/neural-super-sampling",
        filename="nss_v1_0_1_high_fp32.pt",
        revision="main",
    )

    checkpoint = torch.load(
        weights, map_location=torch.device("cpu"), weights_only=True
    )
    state_dict = {
        f"auto_encoder.{key.removeprefix('autoencoder.')}": value
        for key, value in checkpoint["model_state_dict"].items()
    }

    nss_model = NSS()
    nss_model.load_state_dict(state_dict, strict=True)
    return nss_model.auto_encoder


def example_inputs():
    return load_nss_verification_inputs()


def random_inputs():
    return (torch.rand((1, _NSS_INPUT_CHANNELS, 544, 960)),)


input_test_data = REAL_AND_RANDOM_DATA
is_qat_test_data = PTQ_AND_QAT_DATA


def _set_nss_calibration_samples(pipeline):
    return pipeline.set_quantization_calibration(
        iter_nss_test_calibration_samples(),
        dynamic_shapes=_NSS_QUANTIZATION_DYNAMIC_SHAPES,
    )


@common.parametrize("use_real_data", input_test_data)
def test_nss_tosa_FP(use_real_data):
    pipeline = TosaPipelineFP[input_t](
        nss().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    if use_real_data:
        pipeline.add_stage_after("export", pipeline.tester.dump_operator_distribution)
    pipeline.run()


@common.parametrize("is_qat", is_qat_test_data)
@common.parametrize("use_real_data", input_test_data)
def test_nss_tosa_INT(use_real_data, is_qat):
    if is_qat:
        pipeline_kwargs = {
            # Frobenius norm & cosine theshold check disabled for QAT as smoke test has innacurate results and only checks flow functionality.
            "frobenius_threshold": None,
            "cosine_threshold": None,
            "qtol": 12 if use_real_data else 8,
        }
    else:
        pipeline_kwargs = (
            {"frobenius_threshold": 0.32, "qtol": 12} if use_real_data else {"qtol": 7}
        )
    pipeline = TosaPipelineINT[input_t](
        nss().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        is_qat=is_qat,
        **pipeline_kwargs,
    )
    if use_real_data:
        _set_nss_calibration_samples(pipeline)
    pipeline.run()


@pytest.mark.skip(reason="No support for aten_upsample_nearest2d_vec on U55")
@common.XfailIfNoCorstone300
def test_nss_u55_INT():
    pipeline = EthosU55PipelineINT[input_t](
        nss().eval(),
        example_inputs(),
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
        use_to_edge_transform_and_lower=True,
    )
    _set_nss_calibration_samples(pipeline)
    pipeline.run()


@pytest.mark.skip(
    reason="Fails at input memory allocation for input shape: [1, 12, 544, 960]"
)
@common.XfailIfNoCorstone320
def test_nss_u85_INT():
    pipeline = EthosU85PipelineINT[input_t](
        nss().eval(),
        example_inputs(),
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
        use_to_edge_transform_and_lower=True,
    )
    _set_nss_calibration_samples(pipeline)
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("use_real_data", input_test_data)
def test_nss_vgf_FP(use_real_data):
    pipeline = VgfPipeline[input_t](
        nss().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        run_on_vulkan_runtime=True,
        quantize=False,
        # Override tosa version to test FP-only path
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("is_qat", is_qat_test_data)
@common.parametrize("use_real_data", input_test_data)
def test_nss_vgf_INT(use_real_data, is_qat):
    pipeline = VgfPipeline[input_t](
        nss().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
        symmetric_io_quantization=True,
        use_to_edge_transform_and_lower=True,
        run_on_vulkan_runtime=True,
        quantize=True,
        is_qat=is_qat,
        # Override tosa version to test INT-only path
        tosa_version="TOSA-1.0+INT",
        qtol=12 if use_real_data else (8 if is_qat else 7),
    )
    if use_real_data:
        _set_nss_calibration_samples(pipeline)
    pipeline.run()
