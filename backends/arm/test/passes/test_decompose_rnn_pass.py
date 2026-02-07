# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for DecomposeRnnPass.

Note: PyTorch's export may already decompose aten.rnn_tanh.input and
aten.rnn_relu.input into elementary ops before our pass runs. These tests
verify that:
1. The pass runs without errors on RNN models
2. The output is numerically correct after the pass
3. The pass handles various RNN configurations (bidirectional, multi-layer, etc.)
"""

from typing import Tuple

import torch
from executorch.backends.arm._passes.decompose_rnn_pass import DecomposeRnnPass
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.Tensor, torch.Tensor]  # Input x, hidden state h


class RNNTanh(torch.nn.Module):
    """RNN model using tanh nonlinearity"""

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
        self.rnn = torch.nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",
            bias=bias,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.batch_first = batch_first

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output, h_n = self.rnn(x, h)
        return output, h_n

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
        return (x, h)


class RNNRelu(torch.nn.Module):
    """RNN model using relu nonlinearity"""

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
        self.rnn = torch.nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="relu",
            bias=bias,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.batch_first = batch_first

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output, h_n = self.rnn(x, h)
        return output, h_n

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
        return (x, h)


def test_decompose_rnn_tanh_tosa_FP():
    """Test basic RNN with tanh (single layer, unidirectional, with bias)"""
    module = RNNTanh()
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        quantize=False,
        ops_after_pass={
            # RNN tanh: 1 tanh per timestep
            "executorch_exir_dialects_edge__ops_aten_tanh_default": 5,
        },
        pass_list=[DecomposeRnnPass],
    )
    pipeline.run()


def test_decompose_rnn_relu_tosa_FP():
    """Test basic RNN with relu (single layer, unidirectional, with bias)"""
    module = RNNRelu()
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        quantize=False,
        ops_after_pass={
            # RNN relu: 1 relu per timestep
            "executorch_exir_dialects_edge__ops_aten_relu_default": 5,
        },
        pass_list=[DecomposeRnnPass],
    )
    pipeline.run()


def test_decompose_rnn_tanh_bidirectional_tosa_FP():
    """Test bidirectional RNN with tanh"""
    module = RNNTanh(bidirectional=True)
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        quantize=False,
        ops_after_pass={
            # Bidirectional: 2x the ops
            "executorch_exir_dialects_edge__ops_aten_tanh_default": 10,
        },
        pass_list=[DecomposeRnnPass],
    )
    pipeline.run()


def test_decompose_rnn_tanh_no_bias_tosa_FP():
    """Test RNN with tanh without bias"""
    module = RNNTanh(bias=False)
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        quantize=False,
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_tanh_default": 5,
        },
        pass_list=[DecomposeRnnPass],
    )
    pipeline.run()


def test_decompose_rnn_tanh_multilayer_tosa_FP():
    """Test multi-layer RNN with tanh"""
    module = RNNTanh(num_layers=2)
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        quantize=False,
        ops_after_pass={
            # 2 layers: 2x the ops per timestep
            "executorch_exir_dialects_edge__ops_aten_tanh_default": 10,
        },
        pass_list=[DecomposeRnnPass],
    )
    pipeline.run()
