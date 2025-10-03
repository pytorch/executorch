# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

from torch.nn.quantizable.modules import rnn

input_t = Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]  #  (h0, c0)


def get_test_inputs():
    return (
        torch.randn(5, 3, 10),  # input
        (torch.randn(2, 3, 20), torch.randn(2, 3, 20)),  # (h0, c0)
    )


class TestLSTM:
    """Tests quantizable LSTM module."""

    """
    Currently only the quantizable LSTM module has been verified with the arm backend.
    There may be plans to update this to use torch.nn.LSTM.
    TODO: MLETORCH-622
    """
    lstm = rnn.LSTM(10, 20, 2)
    lstm = lstm.eval()

    # Used e.g. for quantization calibration and shape extraction in the tester
    model_example_inputs = get_test_inputs()


def test_lstm_tosa_FP():
    pipeline = TosaPipelineFP[input_t](
        TestLSTM.lstm,
        TestLSTM.model_example_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.change_args("run_method_and_compare_outputs", get_test_inputs(), atol=3e-1)
    pipeline.run()


def test_lstm_tosa_INT():
    pipeline = TosaPipelineINT[input_t](
        TestLSTM.lstm,
        TestLSTM.model_example_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.change_args(
        "run_method_and_compare_outputs", get_test_inputs(), atol=3e-1, qtol=1.0
    )
    pipeline.run()


@common.XfailIfNoCorstone300
def test_lstm_u55_INT():
    pipeline = EthosU55PipelineINT[input_t](
        TestLSTM.lstm,
        TestLSTM.model_example_inputs,
        aten_ops=[],
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.change_args(
        "run_method_and_compare_outputs", get_test_inputs(), atol=3e-1, qtol=1.0
    )
    pipeline.run()


@common.XfailIfNoCorstone320
def test_lstm_u85_INT():
    pipeline = EthosU85PipelineINT[input_t](
        TestLSTM.lstm,
        TestLSTM.model_example_inputs,
        aten_ops=[],
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.change_args(
        "run_method_and_compare_outputs", get_test_inputs(), atol=3e-1, qtol=1.0
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_lstm_vgf_INT():
    pipeline = VgfPipeline[input_t](
        TestLSTM.lstm,
        TestLSTM.model_example_inputs,
        aten_op=[],
        exir_op=[],
        tosa_version="TOSA-1.0+INT",
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_lstm_vgf_FP():
    pipeline = VgfPipeline[input_t](
        TestLSTM.lstm,
        TestLSTM.model_example_inputs,
        aten_op=[],
        exir_op=[],
        tosa_version="TOSA-1.0+FP",
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()
