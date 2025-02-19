# Copyright 2025 Arm Limited and/or its affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pytest

from executorch.backends.arm.test import common, conftest

from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.examples.models import deeplab_v3


class TestDl3(unittest.TestCase):
    """Tests DeepLabv3."""

    dl3 = deeplab_v3.DeepLabV3ResNet50Model()
    model_example_inputs = dl3.get_example_inputs()
    dl3 = dl3.get_eager_model()

    @unittest.expectedFailure
    def test_dl3_tosa_MI(self):
        (
            ArmTester(
                self.dl3,
                example_inputs=self.model_example_inputs,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
            )
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .run_method_and_compare_outputs(inputs=self.dl3.get_example_inputs())
        )

    @unittest.expectedFailure
    def test_dl3_tosa_BI(self):
        (
            ArmTester(
                self.dl3,
                example_inputs=self.model_example_inputs,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+BI"),
            )
            .quantize()
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .run_method_and_compare_outputs(
                atol=1.0, qtol=1, inputs=self.dl3.get_example_inputs()
            )
        )

    @pytest.mark.slow
    @pytest.mark.corstone_fvp
    @unittest.skip
    def test_dl3_u55_BI(self):
        tester = (
            ArmTester(
                self.dl3,
                example_inputs=self.model_example_inputs,
                compile_spec=common.get_u55_compile_spec(),
            )
            .quantize()
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .serialize()
        )
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(
                atol=1.0, qtol=1, inputs=self.dl3.get_example_inputs()
            )

    @pytest.mark.slow
    @pytest.mark.corstone_fvp
    @unittest.skip
    def test_dl3_u85_BI(self):
        tester = (
            ArmTester(
                self.dl3,
                example_inputs=self.model_example_inputs,
                compile_spec=common.get_u85_compile_spec(),
            )
            .quantize()
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .serialize()
        )
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(
                atol=1.0, qtol=1, inputs=self.dl3.get_example_inputs()
            )
