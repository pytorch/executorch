# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for DecomposeGruPass.

Note: PyTorch's export may already decompose aten.gru.input into elementary ops
before our pass runs. These tests verify that:
1. The pass runs without errors on GRU models
2. The output is numerically correct after the pass
3. The pass handles various GRU configurations (bidirectional, multi-layer, etc.)

The DecomposeGruPass is designed to handle cases where GRU survives export,
which can happen with certain export configurations or future PyTorch versions.
"""

from typing import Tuple

import torch
from executorch.backends.arm._passes.decompose_gru_pass import DecomposeGruPass
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.Tensor, torch.Tensor]  # Input x, hidden state h


class GRU(torch.nn.Module):
    """
    Basic GRU model using torch.nn.GRU layer
    """

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


def test_decompose_gru_tosa_FP():
    """Test basic GRU (single layer, unidirectional, with bias)"""
    module = GRU()
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        quantize=False,
        # GRU may already be decomposed by PyTorch export, so we only check
        # that elementary ops are present after the pass and output is correct
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_sigmoid_default": 10,
            "executorch_exir_dialects_edge__ops_aten_tanh_default": 5,
        },
        pass_list=[DecomposeGruPass],
    )
    pipeline.run()


def test_decompose_gru_bidirectional_tosa_FP():
    """Test bidirectional GRU"""
    module = GRU(bidirectional=True)
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        quantize=False,
        ops_after_pass={
            # Bidirectional: 2x the ops (forward + backward)
            "executorch_exir_dialects_edge__ops_aten_sigmoid_default": 20,
            "executorch_exir_dialects_edge__ops_aten_tanh_default": 10,
        },
        pass_list=[DecomposeGruPass],
    )
    pipeline.run()


def test_decompose_gru_no_bias_tosa_FP():
    """Test GRU without bias"""
    module = GRU(bias=False)
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        quantize=False,
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_sigmoid_default": 10,
            "executorch_exir_dialects_edge__ops_aten_tanh_default": 5,
        },
        pass_list=[DecomposeGruPass],
    )
    pipeline.run()


def test_decompose_gru_multilayer_tosa_FP():
    """Test multi-layer GRU"""
    module = GRU(num_layers=2)
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        quantize=False,
        ops_after_pass={
            # 2 layers: 2x the ops per timestep
            "executorch_exir_dialects_edge__ops_aten_sigmoid_default": 20,
            "executorch_exir_dialects_edge__ops_aten_tanh_default": 10,
        },
        pass_list=[DecomposeGruPass],
    )
    pipeline.run()
