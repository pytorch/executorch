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

from executorch.exir import EdgeCompileConfig

from torch.nn.quantizable.modules import rnn


class TestLSTM(unittest.TestCase):
    """Tests LSTM."""

    lstm = rnn.LSTM(10, 20, 2)
    lstm = lstm.eval()

    input_tensor = torch.randn(5, 3, 10)
    h0 = torch.randn(2, 3, 20)
    c0 = torch.randn(2, 3, 20)

    model_inputs = (input_tensor, (h0, c0))

    _edge_compile_config = EdgeCompileConfig(
        _skip_dim_order=True,  # TODO(T182928844): Delegate dim order op to backend.
    )

    def test_lstm_tosa_MI(self):
        (
            ArmTester(
                self.lstm,
                example_inputs=self.model_inputs,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
            )
            .export()
            .to_edge_transform_and_lower(edge_compile_config=self._edge_compile_config)
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
            .to_edge_transform_and_lower(edge_compile_config=self._edge_compile_config)
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
            .to_edge_transform_and_lower(edge_compile_config=self._edge_compile_config)
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
            .to_edge_transform_and_lower(edge_compile_config=self._edge_compile_config)
            .to_executorch()
            .serialize()
        )
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(
                atol=3e-1, qtol=1, inputs=self.model_inputs
            )
