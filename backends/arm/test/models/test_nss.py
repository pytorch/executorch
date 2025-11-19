# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest
import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

from huggingface_hub import hf_hub_download

from ng_model_gym.usecases.nss.model.model_blocks import (  # type: ignore[import-not-found,import-untyped]
    AutoEncoderV1,
)

input_t = Tuple[torch.Tensor]  # Input x


class NSS(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auto_encoder = AutoEncoderV1()


def nss() -> AutoEncoderV1:
    """Get an instance of NSS with weights loaded."""

    weights = hf_hub_download(
        repo_id="Arm/neural-super-sampling", filename="nss_v0.1.0_fp32.pt"
    )

    nss_model = NSS()
    nss_model.load_state_dict(
        torch.load(weights, map_location=torch.device("cpu"), weights_only=True),
        strict=False,
    )
    return nss_model.auto_encoder


def example_inputs():
    return (torch.randn((1, 12, 544, 960)),)


def test_nss_tosa_FP():
    pipeline = TosaPipelineFP[input_t](
        nss().eval(),
        example_inputs(),
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.add_stage_after("export", pipeline.tester.dump_operator_distribution)
    pipeline.run()


def test_nss_tosa_INT():
    pipeline = TosaPipelineINT[input_t](
        nss().eval(),
        example_inputs(),
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
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
    pipeline.run()


@pytest.mark.xfail(
    reason="[MLETORCH-1430]: Double types are not supported in buffers in MSL"
)
@common.SkipIfNoModelConverter
def test_nss_vgf_FP():
    pipeline = VgfPipeline[input_t](
        nss().eval(),
        example_inputs(),
        aten_op=[],
        exir_op=[],
        tosa_version="TOSA-1.0+FP",
        use_to_edge_transform_and_lower=True,
        run_on_vulkan_runtime=True,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_nss_vgf_INT():
    pipeline = VgfPipeline[input_t](
        nss().eval(),
        example_inputs(),
        aten_op=[],
        exir_op=[],
        tosa_version="TOSA-1.0+INT",
        symmetric_io_quantization=True,
        use_to_edge_transform_and_lower=True,
        run_on_vulkan_runtime=True,
    )
    pipeline.run()


ModelUnderTest = nss().eval()
ModelInputs = example_inputs()
