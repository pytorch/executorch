# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import Tuple

import torch
from executorch.backends.arm._passes import BroadcastArgsPass

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.Tensor]  # Input x


class NeedsMultipleBroadcastsModel(torch.nn.Module):
    test_data = (torch.rand(1, 10), torch.rand(10, 1))

    def __init__(self, op: operator):
        self.op = op
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return self.op(x, y)


modules = {
    "add": NeedsMultipleBroadcastsModel(operator.add),
    "sub": NeedsMultipleBroadcastsModel(operator.sub),
    "mul": NeedsMultipleBroadcastsModel(operator.mul),
    "div": NeedsMultipleBroadcastsModel(operator.truediv),
}


@common.parametrize("module", modules)
def test_multiple_broacasts_model(module: NeedsMultipleBroadcastsModel):
    test_data = module.test_data
    ops_not_before_pass = [
        "executorch_exir_dialects_edge__ops_aten_repeat_default",
    ]
    ops_after_pass = {
        "executorch_exir_dialects_edge__ops_aten_repeat_default": 1,
    }
    pipeline = PassPipeline[input_t](
        module,
        test_data,
        quantize=True,
        ops_not_before_pass=ops_not_before_pass,
        ops_after_pass=ops_after_pass,
        pass_list=[BroadcastArgsPass],
    )
    pipeline.run()
