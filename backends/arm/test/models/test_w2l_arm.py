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
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)

from torchaudio import models

input_t = Tuple[torch.Tensor]  # Input x


def get_test_inputs(batch_size, num_features, input_frames):
    return (torch.randn(batch_size, num_features, input_frames),)


class TestW2L(unittest.TestCase):
    """Tests Wav2Letter."""

    batch_size = 10
    input_frames = 400
    num_features = 1

    w2l = models.Wav2Letter(num_features=num_features).eval()
    model_example_inputs = get_test_inputs(batch_size, num_features, input_frames)

    all_operators = [
        "executorch_exir_dialects_edge__ops_aten_convolution_default",
        "executorch_exir_dialects_edge__ops_aten__log_softmax_default",
        "executorch_exir_dialects_edge__ops_aten_relu_default",
    ]


@pytest.mark.slow  # about 3min on std laptop
def test_w2l_tosa_MI():
    pipeline = TosaPipelineMI[input_t](
        TestW2L.w2l,
        TestW2L.model_example_inputs,
        aten_op=[],
        exir_op=TestW2L.all_operators,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@pytest.mark.slow  # about 1min on std laptop
@pytest.mark.flaky
def test_w2l_tosa_BI():
    pipeline = TosaPipelineBI[input_t](
        TestW2L.w2l,
        TestW2L.model_example_inputs,
        aten_op=[],
        exir_op=TestW2L.all_operators,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@pytest.mark.slow
@common.XfailIfNoCorstone300
@pytest.mark.xfail(
    reason="MLETORCH-1009: Wav2Letter fails on U55 due to unsupported conditions",
    strict=False,
)
def test_w2l_u55_BI():
    pipeline = EthosU55PipelineBI[input_t](
        TestW2L.w2l,
        TestW2L.model_example_inputs,
        aten_ops=[],
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
        run_on_fvp=True,
    )
    pipeline.run()


@pytest.mark.slow
@common.XfailIfNoCorstone320
@pytest.mark.skip(reason="Intermittent timeout issue: MLETORCH-856")
def test_w2l_u85_BI():
    pipeline = EthosU85PipelineBI[input_t](
        TestW2L.w2l,
        TestW2L.model_example_inputs,
        aten_ops=[],
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
        run_on_fvp=True,
    )
    pipeline.run()
