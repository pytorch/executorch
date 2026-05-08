# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for DecomposeGruPass."""

from typing import Tuple

import torch
from executorch.backends.arm._passes import DecomposeGruPass
from executorch.backends.arm.test.tester.test_pipeline import (
    PassPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
)

input_t = Tuple[torch.Tensor, torch.Tensor]  # Input x, hidden state h


class GRU(torch.nn.Module):
    """Basic GRU model using torch.nn.GRU layer."""

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
        self.gru = torch.nn.GRU(
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
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output, h_n = self.gru(x, h)
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


chain_input_t = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class SequentialGRU(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru1 = torch.nn.GRU(10, 12, batch_first=True)
        self.gru2 = torch.nn.GRU(12, 8, batch_first=True)

    def forward(
        self, x: torch.Tensor, h1: torch.Tensor, h2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y, _ = self.gru1(x, h1)
        z, h_n = self.gru2(y, h2)
        return z, h_n

    def get_inputs(self) -> chain_input_t:
        return (
            torch.randn(2, 5, 10),
            torch.randn(1, 2, 12),
            torch.randn(1, 2, 8),
        )


def _make_gru_fp_pipeline(module: GRU) -> TosaPipelineFP:
    pipeline = TosaPipelineFP[input_t](
        module,
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


def _make_gru_int_pipeline(module: GRU) -> TosaPipelineINT:
    pipeline = TosaPipelineINT[input_t](
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
        atol=3e-1,
        qtol=1.0,
    )
    return pipeline


def test_decompose_gru_tosa_FP():
    """Test basic GRU (single layer, unidirectional, with bias)."""
    _make_gru_fp_pipeline(GRU()).run()


def test_decompose_gru_tosa_INT():
    """Test basic GRU through quantized pipeline."""
    _make_gru_int_pipeline(GRU()).run()


def test_decompose_gru_tosa_FP_bidirectional():
    """Test bidirectional GRU."""
    _make_gru_fp_pipeline(GRU(bidirectional=True)).run()


def test_decompose_gru_tosa_INT_bidirectional():
    """Test bidirectional GRU through quantized pipeline."""
    _make_gru_int_pipeline(GRU(bidirectional=True)).run()


def test_decompose_gru_tosa_FP_no_bias():
    """Test GRU without bias."""
    _make_gru_fp_pipeline(GRU(bias=False)).run()


def test_decompose_gru_tosa_INT_no_bias():
    """Test GRU without bias through quantized pipeline."""
    _make_gru_int_pipeline(GRU(bias=False)).run()


def test_decompose_gru_tosa_FP_multilayer():
    """Test multi-layer GRU."""
    _make_gru_fp_pipeline(GRU(num_layers=2)).run()


def test_decompose_gru_tosa_INT_multilayer():
    """Test multi-layer GRU through quantized pipeline."""
    _make_gru_int_pipeline(GRU(num_layers=2)).run()


def test_decompose_gru_pass_handles_chained_grus() -> None:
    module = SequentialGRU()
    pipeline = PassPipeline(
        module,
        module.get_inputs(),
        quantize=True,
        pass_list=[DecomposeGruPass],
    )
    pipeline.run()
