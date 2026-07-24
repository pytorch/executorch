# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""End-to-end TosaPipeline tests for recurrent layer decomposition passes.

Tests DecomposeRnnPass, DecomposeGruPass, and DecomposeLstmPass through
the full ARM/TOSA lowering pipeline using TosaPipelineFP (floating point)
and TosaPipelineINT (quantized INT8) to verify that decomposed recurrent
layers compile and run correctly on the TOSA reference model.

These complement the pass-specific unit tests in test_decompose_rnn_pass.py,
test_decompose_gru_pass.py, and test_decompose_lstm_pass.py.
"""

from typing import Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
)


# ──────────────────────────── Input type aliases ────────────────────────────

rnn_input_t = Tuple[torch.Tensor, torch.Tensor]  # (x, h)
lstm_input_t = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # (x, h0, c0)


# ──────────────────────────── Model definitions ─────────────────────────────


class RNNTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.RNN(10, 20, 1, nonlinearity="tanh", batch_first=True)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rnn(x, h)

    @staticmethod
    def get_inputs() -> rnn_input_t:
        return (torch.randn(2, 5, 10), torch.randn(1, 2, 20))


class RNNRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.RNN(10, 20, 1, nonlinearity="relu", batch_first=True)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rnn(x, h)

    @staticmethod
    def get_inputs() -> rnn_input_t:
        return (torch.randn(2, 5, 10), torch.randn(1, 2, 20))


class GRU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = torch.nn.GRU(10, 20, 1, batch_first=True)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.gru(x, h)

    @staticmethod
    def get_inputs() -> rnn_input_t:
        return (torch.randn(2, 5, 10), torch.randn(1, 2, 20))


class LSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(10, 20, 1, batch_first=True)

    def forward(
        self, x: torch.Tensor, h0: torch.Tensor, c0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output, (h_n, c_n) = self.lstm(x, (h0, c0))
        return output, h_n, c_n

    @staticmethod
    def get_inputs() -> lstm_input_t:
        return (
            torch.randn(2, 5, 10),
            torch.randn(1, 2, 20),
            torch.randn(1, 2, 20),
        )


# ──────────────────────────── Helpers ───────────────────────────────────────


def _run_tosa_fp_pipeline(module, inputs, input_type):
    """Run TosaPipelineFP for a recurrent model."""
    pipeline = TosaPipelineFP[input_type](
        module,
        inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        atol=3e-1,
        rtol=3e-1,
    )
    pipeline.pop_stage("check.aten")
    pipeline.pop_stage("check_count.exir")
    pipeline.run()


def _run_tosa_int_pipeline(module, inputs, input_type):
    """Run TosaPipelineINT for a recurrent model."""
    pipeline = TosaPipelineINT[input_type](
        module,
        inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        frobenius_threshold=None,
        cosine_threshold=None,
    )
    pipeline.pop_stage("check.aten")
    pipeline.pop_stage("check_count.exir")
    if pipeline.has_stage("check.quant_nodes"):
        pipeline.pop_stage("check.quant_nodes")
    if pipeline.has_stage("check_not.quant_nodes"):
        pipeline.pop_stage("check_not.quant_nodes")
    pipeline.change_args(
        "run_method_and_compare_outputs",
        atol=3e-1,
        qtol=1.0,
    )
    pipeline.run()


def _run_ethos_u_pipeline(pipeline_cls, module, inputs, input_type):
    """Run an EthosU pipeline for a recurrent model."""
    pipeline = pipeline_cls[input_type](
        module,
        inputs,
        aten_ops=[],
        exir_ops=[],
    )
    pipeline.pop_stage("check.aten")
    pipeline.pop_stage("check_count.exir")
    if pipeline.has_stage("check.quant_nodes"):
        pipeline.pop_stage("check.quant_nodes")
    if pipeline.has_stage("check_not.quant_nodes"):
        pipeline.pop_stage("check_not.quant_nodes")
    pipeline.run()


# ──────────────────────────── RNN FP tests ──────────────────────────────────


def test_decompose_rnn_tosa_FP_tanh_e2e():
    _run_tosa_fp_pipeline(RNNTanh(), RNNTanh.get_inputs(), rnn_input_t)


def test_decompose_rnn_tosa_FP_relu_e2e():
    _run_tosa_fp_pipeline(RNNRelu(), RNNRelu.get_inputs(), rnn_input_t)


# ──────────────────────────── RNN INT tests ─────────────────────────────────


def test_decompose_rnn_tosa_INT_tanh_e2e():
    _run_tosa_int_pipeline(RNNTanh(), RNNTanh.get_inputs(), rnn_input_t)


def test_decompose_rnn_tosa_INT_relu_e2e():
    _run_tosa_int_pipeline(RNNRelu(), RNNRelu.get_inputs(), rnn_input_t)


# ──────────────────────────── GRU tests ─────────────────────────────────────


def test_decompose_gru_tosa_FP_e2e():
    _run_tosa_fp_pipeline(GRU(), GRU.get_inputs(), rnn_input_t)


def test_decompose_gru_tosa_INT_e2e():
    _run_tosa_int_pipeline(GRU(), GRU.get_inputs(), rnn_input_t)


# ──────────────────────────── LSTM tests ────────────────────────────────────


def test_decompose_lstm_tosa_FP_e2e():
    _run_tosa_fp_pipeline(LSTM(), LSTM.get_inputs(), lstm_input_t)


def test_decompose_lstm_tosa_INT_e2e():
    _run_tosa_int_pipeline(LSTM(), LSTM.get_inputs(), lstm_input_t)


# ──────────────────────── EthosU55 INT probes ───────────────────────────────


@common.XfailIfNoCorstone300
def test_decompose_rnn_u55_INT_tanh_e2e():
    _run_ethos_u_pipeline(
        EthosU55PipelineINT, RNNTanh(), RNNTanh.get_inputs(), rnn_input_t
    )


# ──────────────────────── EthosU85 INT probes ───────────────────────────────


@common.XfailIfNoCorstone320
def test_decompose_rnn_u85_INT_tanh_e2e():
    _run_ethos_u_pipeline(
        EthosU85PipelineINT, RNNTanh(), RNNTanh.get_inputs(), rnn_input_t
    )
