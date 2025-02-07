# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch

from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.arm_tester import ArmTester

from torch.nn.quantizable.modules import rnn


class TestLSTM(unittest.TestCase):
    """Tests quantizable LSTM module."""

    """
    Currently only the quantizable LSTM module has been verified with the arm backend.
    There may be plans to update this to use torch.nn.LSTM.
    TODO: MLETORCH-622
    """
    lstm = rnn.LSTM(10, 20, 2)
    lstm = lstm.eval()

    input_tensor = torch.randn(5, 3, 10)
    h0 = torch.randn(2, 3, 20)
    c0 = torch.randn(2, 3, 20)

    model_inputs = (input_tensor, (h0, c0))

    def test_lstm_tosa_MI(self):
        (
            ArmTester(
                self.lstm,
                example_inputs=self.model_inputs,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
            )
            .export()
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=self.model_inputs)
        )

    def test_lstm_tosa_BI(self):
        (
            ArmTester(
                self.lstm,
                example_inputs=self.model_inputs,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+BI"),
            )
            .quantize()
            .export()
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(atol=3e-1, qtol=1, inputs=self.model_inputs)
        )

    def test_lstm_u55_BI(self):
        tester = (
            ArmTester(
                self.lstm,
                example_inputs=self.model_inputs,
                compile_spec=common.get_u55_compile_spec(),
            )
            .quantize()
            .export()
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
        )
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(
                atol=3e-1, qtol=1, inputs=self.model_inputs
            )

    def test_lstm_u85_BI(self):
        tester = (
            ArmTester(
                self.lstm,
                example_inputs=self.model_inputs,
                compile_spec=common.get_u85_compile_spec(),
            )
            .quantize()
            .export()
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
        )
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(
                atol=3e-1, qtol=1, inputs=self.model_inputs
            )
