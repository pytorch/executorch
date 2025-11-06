# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
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

from torchaudio import models  # type: ignore[import-untyped]

input_t = Tuple[torch.Tensor]  # Input x


def get_test_inputs(batch_size, num_features, input_frames):
    return (torch.randn(batch_size, num_features, input_frames),)


class TestW2L(unittest.TestCase):
    """Tests Wav2Letter."""

    batch_size = 10
    input_frames = 400
    num_features = 1

    model_example_inputs = get_test_inputs(batch_size, num_features, input_frames)

    all_operators = [
        "executorch_exir_dialects_edge__ops_aten_convolution_default",
        "executorch_exir_dialects_edge__ops_aten__log_softmax_default",
        "executorch_exir_dialects_edge__ops_aten_relu_default",
    ]

    @staticmethod
    def create_model(input_type: str = "waveform"):
        return models.Wav2Letter(
            num_features=TestW2L.num_features, input_type=input_type
        ).eval()


@pytest.mark.slow  # about 3min on std laptop
def test_w2l_tosa_FP():
    pipeline = TosaPipelineFP[input_t](
        TestW2L.create_model(),
        TestW2L.model_example_inputs,
        aten_op=[],
        exir_op=TestW2L.all_operators,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@pytest.mark.slow  # about 1min on std laptop
@pytest.mark.flaky
def test_w2l_tosa_INT():
    pipeline = TosaPipelineINT[input_t](
        TestW2L.create_model(),
        TestW2L.model_example_inputs,
        aten_op=[],
        exir_op=TestW2L.all_operators,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@pytest.mark.slow
@common.XfailIfNoCorstone300
@pytest.mark.xfail(
    reason="Wav2Letter fails on U55 due to insufficient memory",
    strict=True,
)
def test_w2l_u55_INT():
    pipeline = EthosU55PipelineINT[input_t](
        # Use "power_spectrum" variant because the default ("waveform") has a
        # conv1d layer with an unsupported stride size.
        TestW2L.create_model("power_spectrum"),
        TestW2L.model_example_inputs,
        aten_ops=[],
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@pytest.mark.slow
@common.XfailIfNoCorstone320
@pytest.mark.skip(reason="Intermittent timeout issue: MLETORCH-856")
def test_w2l_u85_INT():
    pipeline = EthosU85PipelineINT[input_t](
        TestW2L.create_model(),
        TestW2L.model_example_inputs,
        aten_ops=[],
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@pytest.mark.slow
def test_w2l_vgf_INT():
    pipeline = VgfPipeline[input_t](
        TestW2L.create_model(),
        TestW2L.model_example_inputs,
        aten_op=[],
        exir_op=TestW2L.all_operators,
        tosa_version="TOSA-1.0+INT",
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_w2l_vgf_FP():
    pipeline = VgfPipeline[input_t](
        TestW2L.create_model(),
        TestW2L.model_example_inputs,
        aten_op=[],
        exir_op=TestW2L.all_operators,
        tosa_version="TOSA-1.0+FP",
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()
