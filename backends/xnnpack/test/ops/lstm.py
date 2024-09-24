# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

from executorch.backends.xnnpack.test.tester import Tester
from executorch.backends.xnnpack.test.tester.tester import ToEdgeTransformAndLower


class TestLSTM(unittest.TestCase):
    class LSTMLinear(torch.nn.Module):
        def __init__(self, input_size, hidden_size, out_size):
            super().__init__()
            self.lstm = torch.nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, batch_first=True
            )
            self.linear = torch.nn.Linear(hidden_size, hidden_size)
            self.linear2 = torch.nn.Linear(hidden_size, out_size)

        def forward(self, x):
            x, hs = self.lstm(x)
            x = self.linear(x[:, -1, :])
            x = self.linear2(x)
            return torch.nn.functional.log_softmax(x, dim=1)

    def test_fp32_lstm(self):
        (
            Tester(self.LSTMLinear(32, 32, 10), (torch.rand(1, 32, 32),))
            .export()
            .to_edge_transform_and_lower()
            .check_not(["executorch_exir_dialects_edge__ops_aten_addmm_default"])
            .check_not(
                ["p_lstm_weight", "p_lstm_bias"]
            )  # These Should be Consumed by Delegate
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp32_lstm_force_dynamic_linear(self):
        (
            Tester(self.LSTMLinear(32, 32, 10), (torch.rand(1, 32, 32),))
            .export()
            .to_edge_transform_and_lower(
                ToEdgeTransformAndLower(
                    partitioners=[XnnpackPartitioner(force_fp32_dynamic_linear=True)]
                )
            )
            .check_not(["executorch_exir_dialects_edge__ops_aten_addmm_default"])
            # Weights are supplied as input to linears
            .check(["p_lstm_weight_hh_l0", "p_lstm_weight_ih_l0"])
            # Biases are owned by delegates
            .check_not(["p_lstm_bias"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )
