# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest
from typing import Tuple

import pytest

import torch
from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.arm_tester import ArmTester

from executorch.exir.backend.compile_spec_schema import CompileSpec
from torchaudio import models


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_test_inputs(batch_size, num_features, input_frames):
    return (torch.randn(batch_size, num_features, input_frames),)


class TestW2L(unittest.TestCase):
    """Tests Wav2Letter."""

    batch_size = 10
    input_frames = 400
    num_features = 1

    w2l = models.Wav2Letter(num_features=num_features).eval()
    model_example_inputs = get_test_inputs(batch_size, num_features, input_frames)

    all_operators = {
        "executorch_exir_dialects_edge__ops_aten_convolution_default",
        "executorch_exir_dialects_edge__ops_aten__log_softmax_default",
        "executorch_exir_dialects_edge__ops_aten_relu_default",
    }

    operators_after_quantization = all_operators - {
        "executorch_exir_dialects_edge__ops_aten__log_softmax_default",
    }

    @pytest.mark.slow  # about 3min on std laptop
    def test_w2l_tosa_MI(self):
        (
            ArmTester(
                self.w2l,
                example_inputs=self.model_example_inputs,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
            )
            .export()
            .dump_operator_distribution()
            .to_edge_transform_and_lower()
            .dump_operator_distribution()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(
                inputs=get_test_inputs(
                    self.batch_size, self.num_features, self.input_frames
                )
            )
        )

    @pytest.mark.slow  # about 1min on std laptop
    def test_w2l_tosa_BI(self):
        (
            ArmTester(
                self.w2l,
                example_inputs=self.model_example_inputs,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+BI"),
            )
            .quantize()
            .export()
            .dump_operator_distribution()
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(
                atol=0.1,
                qtol=1,
                inputs=get_test_inputs(
                    self.batch_size, self.num_features, self.input_frames
                ),
            )
        )

    def _test_w2l_ethos_BI_pipeline(
        self,
        module: torch.nn.Module,
        test_data: Tuple[torch.Tensor],
        compile_spec: CompileSpec,
    ):
        tester = (
            ArmTester(module, example_inputs=test_data, compile_spec=compile_spec)
            .quantize()
            .export()
            .to_edge()
            .check(list(self.operators_after_quantization))
            .partition()
            .to_executorch()
            .serialize()
        )
        return tester

    # TODO: expected fail as TOSA.Transpose is not supported by Ethos-U55
    @pytest.mark.slow
    @pytest.mark.corstone_fvp
    @conftest.expectedFailureOnFVP
    def test_w2l_u55_BI(self):
        tester = self._test_w2l_ethos_BI_pipeline(
            self.w2l,
            self.model_example_inputs,
            common.get_u55_compile_spec(),
        )

        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(
                atol=1.0,
                qtol=1,
                inputs=get_test_inputs(
                    self.batch_size, self.num_features, self.input_frames
                ),
            )

    @pytest.mark.slow
    @pytest.mark.corstone_fvp
    @conftest.expectedFailureOnFVP  # TODO: MLETORCH-761
    def test_w2l_u85_BI(self):
        tester = self._test_w2l_ethos_BI_pipeline(
            self.w2l,
            self.model_example_inputs,
            common.get_u85_compile_spec(),
        )

        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(
                atol=1.0,
                qtol=1,
                inputs=get_test_inputs(
                    self.batch_size, self.num_features, self.input_frames
                ),
            )
