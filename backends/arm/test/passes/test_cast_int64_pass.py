# Copyright 2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.arm._passes.cast_int64_pass import CastInt64ToInt32Pass

from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.arm_tester import ArmTester, RunPasses


class Int64Model(torch.nn.Module):

    def forward(self, x: torch.Tensor):
        return x + 3

    def get_inputs(self):
        return (torch.rand(4),)


class TestCastInt64Pass(unittest.TestCase):

    def test_int64_model(self):
        module = Int64Model()
        test_pass_stage = RunPasses(passes_with_exported_program=[CastInt64ToInt32Pass])
        tester = (
            ArmTester(
                module,
                example_inputs=module.get_inputs(),
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+BI"),
            )
            .export()
            .to_edge()
            .run_passes(test_pass_stage)
            .run_method_and_compare_outputs()
        )
        exported_program = tester.get_artifact("RunPasses").exported_program()
        for state in exported_program.state_dict:
            assert exported_program.state_dict[state].dtype != torch.int64
