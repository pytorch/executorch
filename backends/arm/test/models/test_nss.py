# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from typing import Tuple

import pytest
import torch
from executorch.backends.arm.scripts.neural_graphics_test_data import (
    iter_calibration_samples,
    load_verification_inputs,
    nss_test_calibration_path,
)

from executorch.backends.arm.test import common
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

_RELEASE_REFS = (
    os.environ.get("GITHUB_REF", ""),
    os.environ.get("GITHUB_REF_NAME", ""),
    os.environ.get("GITHUB_BASE_REF", ""),
)
_IS_FROZEN_RELEASE = any(
    ref.removeprefix("refs/heads/").startswith("release/") for ref in _RELEASE_REFS
)
pytestmark = pytest.mark.skipif(
    _IS_FROZEN_RELEASE,
    reason="NSS tests depend on resources fetched from main.",
)

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
    return load_verification_inputs()


def random_inputs():
    return (torch.rand((1, 12, 544, 960)),)


input_test_data = {
    "real_data": True,
    "random_data": False,
}


def _nss_calibration_path() -> Path:
    path = nss_test_calibration_path()
    if not path.exists():
        raise RuntimeError(
            "NSS calibration data is prepared by "
            "backends/arm/scripts/install_models_for_test.sh."
        )
    return path


def _set_nss_calibration_samples(pipeline):
    quantize_stage = pipeline._stages[pipeline.find_pos("quantize")].args[0]
    quantize_stage.dynamic_shapes = _NSS_QUANTIZATION_DYNAMIC_SHAPES
    quantize_stage.calibration_samples = iter_calibration_samples(
        _nss_calibration_path(), num_samples=3663
    )
    return pipeline


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


@common.parametrize("use_real_data", input_test_data)
def test_nss_tosa_INT(use_real_data):
    pipeline_kwargs = (
        {"frobenius_threshold": 0.32, "qtol": 12} if use_real_data else {"qtol": 7}
    )
    pipeline = TosaPipelineINT[input_t](
        nss().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
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
@common.parametrize("use_real_data", input_test_data)
def test_nss_vgf_INT(use_real_data):
    pipeline = VgfPipeline[input_t](
        nss().eval(),
        example_inputs() if use_real_data else random_inputs(),
        aten_op=[],
        exir_op=[],
        symmetric_io_quantization=True,
        use_to_edge_transform_and_lower=True,
        run_on_vulkan_runtime=True,
        quantize=True,
        # Override tosa version to test INT-only path
        tosa_version="TOSA-1.0+INT",
        qtol=12 if use_real_data else 7,
    )
    if use_real_data:
        _set_nss_calibration_samples(pipeline)
    pipeline.run()
