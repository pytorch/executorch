# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm._passes.convert_expand_copy_to_repeat import (
    ConvertExpandCopyToRepeatPass,
)

from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.Tensor]  # Input x


class Expand(torch.nn.Module):
    """
    Basic expand model using torch.Tensor.expand function
    """

    def __init__(self):
        super(Expand, self).__init__()

    def forward(self, x):
        return x.expand(3, 4)

    def get_inputs(self) -> input_t:
        return (torch.rand(3, 1),)


def test_expand_to_repeat_tosa_INT():
    module = Expand()
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        quantize=True,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_expand_copy_default": 1,
        },
        ops_not_before_pass=["executorch_exir_dialects_edge__ops_aten_repeat_default"],
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_repeat_default": 1,
        },
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_aten_expand_copy_default"
        ],
        pass_list=[ConvertExpandCopyToRepeatPass],
    )
    pipeline.run()
