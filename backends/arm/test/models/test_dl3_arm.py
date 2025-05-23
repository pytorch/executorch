# Copyright 2025 Arm Limited and/or its affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest

import torch

from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)

from executorch.examples.models import deeplab_v3

input_t = Tuple[torch.Tensor]  # Input x


class TestDl3:
    """Tests DeepLabv3."""

    dl3 = deeplab_v3.DeepLabV3ResNet50Model()
    model_example_inputs = dl3.get_example_inputs()
    dl3 = dl3.get_eager_model()


def test_dl3_tosa_MI():
    pipeline = TosaPipelineMI[input_t](
        TestDl3.dl3,
        TestDl3.model_example_inputs,
        aten_op=[],
        exir_op=[],
    )
    pipeline.change_args(
        "run_method_and_compare_outputs", rtol=1.0, atol=1.0
    )  # TODO: MLETORCH-1036 decrease tolerance
    pipeline.run()


def test_dl3_tosa_BI():
    pipeline = TosaPipelineBI[input_t](
        TestDl3.dl3,
        TestDl3.model_example_inputs,
        aten_op=[],
        exir_op=[],
    )
    pipeline.change_args(
        "run_method_and_compare_outputs", rtol=1.0, atol=1.0
    )  # TODO: MLETORCH-1036 decrease tolerance
    pipeline.run()


@common.XfailIfNoCorstone300
@pytest.mark.skip(reason="upsample_bilinear2d operator is not supported on U55")
def test_dl3_u55_BI():
    pipeline = EthosU55PipelineBI[input_t](
        TestDl3.dl3,
        TestDl3.model_example_inputs,
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.change_args(
        "run_method_and_compare_outputs", rtol=1.0, atol=1.0
    )  # TODO: MLETORCH-1036 decrease tolerance
    pipeline.run()


@common.XfailIfNoCorstone320
@pytest.mark.skip(reason="Runs out of memory on U85")
def test_dl3_u85_BI():
    pipeline = EthosU85PipelineBI[input_t](
        TestDl3.dl3,
        TestDl3.model_example_inputs,
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.change_args(
        "run_method_and_compare_outputs", rtol=1.0, atol=1.0
    )  # TODO: MLETORCH-1036 decrease tolerance
    pipeline.run()
