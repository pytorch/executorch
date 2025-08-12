# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Tuple

import torch
from executorch.backends.arm._passes import UnsqueezeBeforeRepeatPass
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[
    torch.Tensor, Dict[str, int], list[str]
]  # Input x, ops_after_pass, ops_not_after_pass


class Repeat(torch.nn.Module):
    """
    Basic repeat model.
    """

    def forward(self, x: torch.Tensor):
        return x.repeat(2, 2, 2, 2)

    test_data: Dict[str, input_t] = {
        "insert_view": (
            (torch.rand((2, 3, 4)),),
            {"aten_repeat_default": 3, "aten_view_copy_default": 4},
            [],
        ),
        "dont_insert_view": (
            (torch.rand((2, 3, 4, 1)),),
            {"aten_repeat_default": 3},
            ["aten_view_copy_default"],
        ),
    }


@common.parametrize("test_data", Repeat.test_data)
def test_unsqueeze_before_repeat_tosa_FP(test_data: input_t):
    """
    When rank(input) != number of repeated dimensions (=4 in Repeat module),
    insert view.
    """
    module = Repeat()
    data, ops_after_pass, ops_not_after_pass = test_data
    pipeline = PassPipeline(
        module,
        data,
        quantize=False,
        ops_before_pass={"aten_repeat_default": 3},
        ops_not_before_pass=["aten_view_copy_default"],
        ops_after_pass=ops_after_pass,
        ops_not_after_pass=ops_not_after_pass,
        pass_list=[UnsqueezeBeforeRepeatPass],
    )
    pipeline.pop_stage(-1)  # Do not compare output
    pipeline.run()
