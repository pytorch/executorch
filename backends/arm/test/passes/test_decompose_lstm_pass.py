# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for DecomposeLstmPass."""

from typing import Tuple

import pytest
import torch
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
)

lstm_input_t = Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


class LSTMModel(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 20,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

    def forward(
        self, x: torch.Tensor, hx: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output, (h_n, c_n) = self.lstm(x, hx)
        return output, h_n, c_n

    def get_inputs(self) -> lstm_input_t:
        batch_size = 2
        seq_len = 5
        x = torch.randn(batch_size, seq_len, self.lstm.input_size)
        h = torch.randn(
            self.num_layers * self.num_directions, batch_size, self.hidden_size
        )
        c = torch.randn(
            self.num_layers * self.num_directions, batch_size, self.hidden_size
        )
        return (x, (h, c))


def _run_fp_test(module):
    pipeline = TosaPipelineFP[lstm_input_t](
        module,
        module.get_inputs(),
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.pop_stage("check.aten")
    pipeline.pop_stage("check_count.exir")
    pipeline.change_args(
        "run_method_and_compare_outputs",
        inputs=module.get_inputs(),
        atol=3e-1,
    )
    pipeline.run()


def _run_int_test(module):
    pipeline = TosaPipelineINT[lstm_input_t](
        module,
        module.get_inputs(),
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
        inputs=module.get_inputs(),
        atol=3e-1,
        qtol=1.0,
    )
    pipeline.run()


# ── TosaPipelineFP tests ─────────────────────────────────────────────────────
# DecomposeLstmPass only handles the TOSA INT path. For FP, ExecuTorch's
# get_decompositions() decomposes aten.lstm.input before our backend is
# reached, producing h_n/c_n metadata with incorrect shapes (4D vs 3D).


@pytest.mark.skip(
    reason="Shape mismatch: LSTM FP decomposed by ExecuTorch before backend"
)
def test_decompose_lstm_tosa_FP():
    _run_fp_test(LSTMModel())


@pytest.mark.skip(
    reason="Shape mismatch: LSTM FP decomposed by ExecuTorch before backend"
)
def test_decompose_lstm_tosa_FP_bidirectional():
    _run_fp_test(LSTMModel(bidirectional=True))


@pytest.mark.skip(
    reason="Shape mismatch: LSTM FP decomposed by ExecuTorch before backend"
)
def test_decompose_lstm_tosa_FP_no_bias():
    _run_fp_test(LSTMModel(bias=False))


@pytest.mark.skip(
    reason="Shape mismatch: LSTM FP decomposed by ExecuTorch before backend"
)
def test_decompose_lstm_tosa_FP_multilayer():
    _run_fp_test(LSTMModel(num_layers=2))


# ── TosaPipelineINT tests ────────────────────────────────────────────────────


def test_decompose_lstm_tosa_INT():
    _run_int_test(LSTMModel())


def test_decompose_lstm_tosa_INT_bidirectional():
    _run_int_test(LSTMModel(bidirectional=True))


def test_decompose_lstm_tosa_INT_no_bias():
    _run_int_test(LSTMModel(bias=False))


def test_decompose_lstm_tosa_INT_multilayer():
    _run_int_test(LSTMModel(num_layers=2))
