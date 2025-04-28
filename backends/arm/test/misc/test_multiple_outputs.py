# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pytest
import torch
from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.exir.backend.compile_spec_schema import CompileSpec


class TestMultipleOutputs(unittest.TestCase):
    class MultipleOutputsModule(torch.nn.Module):
        inputs = (torch.randn(10, 4, 5), torch.randn(10, 4, 5))

        def get_inputs(self):
            return self.inputs

        def forward(self, x: torch.Tensor, y: torch.Tensor):
            return (x * y, x.sum(dim=-1, keepdim=True))

    def test_tosa_MI_pipeline(self):
        module = self.MultipleOutputsModule()
        inputs = module.get_inputs()
        (
            ArmTester(
                module,
                example_inputs=inputs,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
            )
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .run_method_and_compare_outputs(inputs=inputs)
        )

    def test_tosa_BI_pipeline(self):
        module = self.MultipleOutputsModule()
        inputs = module.get_inputs()
        (
            ArmTester(
                module,
                example_inputs=inputs,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+BI"),
            )
            .quantize()
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .run_method_and_compare_outputs(inputs=inputs, qtol=1.0)
        )

    def _test_ethosu_BI_pipeline(
        self,
        module: torch.nn.Module,
        test_data: tuple[torch.Tensor],
        compile_spec: CompileSpec,
    ):
        tester = (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=compile_spec,
            )
            .quantize()
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .serialize()
        )
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(qtol=1, inputs=test_data)

    @pytest.mark.corstone_fvp
    def test_u55_BI(self):
        module = self.MultipleOutputsModule()
        test_data = module.get_inputs()
        self._test_ethosu_BI_pipeline(
            module,
            test_data,
            common.get_u55_compile_spec(),
        )

    @pytest.mark.corstone_fvp
    def test_u85_BI(self):
        module = self.MultipleOutputsModule()
        test_data = module.get_inputs()
        self._test_ethosu_BI_pipeline(
            module,
            test_data,
            common.get_u85_compile_spec(),
        )
