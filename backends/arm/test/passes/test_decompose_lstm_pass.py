# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for DecomposeLstmPass.

Note: PyTorch's export may already decompose aten.lstm.input into elementary
ops before our pass runs. These tests verify that:
1. The pass runs without errors on LSTM models
2. The output is numerically correct after the pass
3. The pass handles various LSTM configurations (bidirectional, multi-layer, etc.)
"""

from typing import Tuple

import torch
from executorch.backends.arm._passes.decompose_lstm_pass import DecomposeLstmPass
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

# Input x, (hidden state h, cell state c)
input_t = Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


class LSTM(torch.nn.Module):
    """Basic LSTM model using torch.nn.LSTM layer"""

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
        self.batch_first = batch_first

    def forward(
        self, x: torch.Tensor, hc: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        output, (h_n, c_n) = self.lstm(x, hc)
        return output, (h_n, c_n)

    def get_inputs(self) -> input_t:
        batch_size = 2
        seq_len = 5
        input_size = 10
        if self.batch_first:
            x = torch.randn(batch_size, seq_len, input_size)
        else:
            x = torch.randn(seq_len, batch_size, input_size)
        h = torch.randn(
            self.num_layers * self.num_directions, batch_size, self.hidden_size
        )
        c = torch.randn(
            self.num_layers * self.num_directions, batch_size, self.hidden_size
        )
        return (x, (h, c))


def test_decompose_lstm_tosa_FP():
    """Test basic LSTM (single layer, unidirectional, with bias)"""
    module = LSTM()
    pipeline = PassPipeline[
        input_t
    ](
        module,
        module.get_inputs(),
        quantize=False,
        # LSTM has 4 gates: i, f, g, o
        # i, f, o use sigmoid (3 per timestep)
        # g uses tanh (1 per timestep)
        # Also tanh(c_t) for output (1 per timestep)
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_sigmoid_default": 15,  # 3 * 5 timesteps
            "executorch_exir_dialects_edge__ops_aten_tanh_default": 10,  # 2 * 5 timesteps
        },
        pass_list=[DecomposeLstmPass],
    )
    pipeline.run()


def test_decompose_lstm_bidirectional_tosa_FP():
    """Test bidirectional LSTM"""
    module = LSTM(bidirectional=True)
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        quantize=False,
        ops_after_pass={
            # Bidirectional: 2x the ops
            "executorch_exir_dialects_edge__ops_aten_sigmoid_default": 30,
            "executorch_exir_dialects_edge__ops_aten_tanh_default": 20,
        },
        pass_list=[DecomposeLstmPass],
    )
    pipeline.run()


def test_decompose_lstm_no_bias_tosa_FP():
    """Test LSTM without bias"""
    module = LSTM(bias=False)
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        quantize=False,
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_sigmoid_default": 15,
            "executorch_exir_dialects_edge__ops_aten_tanh_default": 10,
        },
        pass_list=[DecomposeLstmPass],
    )
    pipeline.run()


def test_decompose_lstm_multilayer_tosa_FP():
    """Test multi-layer LSTM"""
    module = LSTM(num_layers=2)
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        quantize=False,
        ops_after_pass={
            # 2 layers: 2x the ops
            "executorch_exir_dialects_edge__ops_aten_sigmoid_default": 30,
            "executorch_exir_dialects_edge__ops_aten_tanh_default": 20,
        },
        pass_list=[DecomposeLstmPass],
    )
    pipeline.run()
