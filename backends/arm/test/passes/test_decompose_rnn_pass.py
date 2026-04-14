# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for DecomposeRnnPass."""

from typing import cast, Protocol, Tuple

import torch
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
)

input_t = Tuple[torch.Tensor, torch.Tensor]  # Input x, hidden state h


class ModuleWithInputs(Protocol):
    def get_inputs(self) -> input_t: ...


class RNNTanh(torch.nn.Module):
    """RNN model using tanh nonlinearity."""

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
    """RNN model using relu nonlinearity."""

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


def _make_rnn_fp_pipeline(module: ModuleWithInputs) -> TosaPipelineFP:
    nn_module = cast(torch.nn.Module, module)
    pipeline = TosaPipelineFP[input_t](
        nn_module,
        module.get_inputs(),
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        atol=3e-1,
        rtol=3e-1,
    )
    pipeline.pop_stage("check.aten")
    pipeline.pop_stage("check_count.exir")
    return pipeline


def _make_rnn_int_pipeline(module: ModuleWithInputs) -> TosaPipelineINT:
    nn_module = cast(torch.nn.Module, module)
    pipeline = TosaPipelineINT[input_t](
        nn_module,
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
        atol=3e-1,
        qtol=1.0,
    )
    return pipeline


def test_decompose_rnn_tosa_FP_tanh():
    """Test basic RNN with tanh (single layer, unidirectional, with bias)."""
    _make_rnn_fp_pipeline(RNNTanh()).run()


def test_decompose_rnn_tosa_INT_tanh():
    """Test basic RNN with tanh through quantized pipeline."""
    _make_rnn_int_pipeline(RNNTanh()).run()


def test_decompose_rnn_tosa_FP_relu():
    """Test basic RNN with relu (single layer, unidirectional, with bias)."""
    _make_rnn_fp_pipeline(RNNRelu()).run()


def test_decompose_rnn_tosa_INT_relu():
    """Test basic RNN with relu through quantized pipeline."""
    _make_rnn_int_pipeline(RNNRelu()).run()


def test_decompose_rnn_tosa_FP_tanh_bidirectional():
    """Test bidirectional RNN with tanh."""
    _make_rnn_fp_pipeline(RNNTanh(bidirectional=True)).run()


def test_decompose_rnn_tosa_INT_tanh_bidirectional():
    """Test bidirectional RNN with tanh through quantized pipeline."""
    _make_rnn_int_pipeline(RNNTanh(bidirectional=True)).run()


def test_decompose_rnn_tosa_FP_tanh_no_bias():
    """Test RNN with tanh without bias."""
    _make_rnn_fp_pipeline(RNNTanh(bias=False)).run()


def test_decompose_rnn_tosa_INT_tanh_no_bias():
    """Test RNN with tanh without bias through quantized pipeline."""
    _make_rnn_int_pipeline(RNNTanh(bias=False)).run()


def test_decompose_rnn_tosa_FP_tanh_multilayer():
    """Test multi-layer RNN with tanh."""
    _make_rnn_fp_pipeline(RNNTanh(num_layers=2)).run()


def test_decompose_rnn_tosa_INT_tanh_multilayer():
    """Test multi-layer RNN with tanh through quantized pipeline."""
    _make_rnn_int_pipeline(RNNTanh(num_layers=2)).run()
