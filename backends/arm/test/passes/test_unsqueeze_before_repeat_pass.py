# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import torch
from executorch.backends.arm._passes.unsqueeze_before_repeat_pass import (
    UnsqueezeBeforeRepeatPass,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.backends.xnnpack.test.tester.tester import RunPasses


class Repeat(torch.nn.Module):
    """
    Basic repeat model.
    """

    def forward(self, x: torch.Tensor):
        return x.repeat(2, 2, 2, 2)


class TestUnsqueezeBeforeRepeatPass(unittest.TestCase):
    def test_tosa_MI_insert_view(self):
        """
        When rank(input) != number of repeated dimensions (=4 in Repeat module),
        insert view.
        """
        module = Repeat()
        inputs = (torch.rand((2, 3, 4)),)
        test_pass_stage = RunPasses([UnsqueezeBeforeRepeatPass])
        (
            (
                ArmTester(
                    module,
                    example_inputs=inputs,
                    compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+MI"),
                )
                .export()
                .to_edge()
                .check(["aten_repeat_default"])
                .check_not(["aten_view_copy_default"])
                .run_passes(test_pass_stage)
                .check(["aten_repeat_default", "aten_view_copy_default"])
            )
        )

    def test_tosa_MI_dont_insert_view(self):
        """
        When rank(input) == number of repeated dimensions (=4 in Repeat module),
        DON'T insert view.
        """
        module = Repeat()
        inputs = (torch.rand((2, 3, 4, 1)),)
        test_pass_stage = RunPasses([UnsqueezeBeforeRepeatPass])
        (
            (
                ArmTester(
                    module,
                    example_inputs=inputs,
                    compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+MI"),
                )
                .export()
                .to_edge()
                .check(["aten_repeat_default"])
                .check_not(["aten_view_copy_default"])
                .run_passes(test_pass_stage)
                .check(["aten_repeat_default"])
                .check_not(["aten_view_copy_default"])
            )
        )
