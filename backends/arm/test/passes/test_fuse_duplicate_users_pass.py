# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm._passes import FuseDuplicateUsersPass
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.Tensor]  # Input x


class FuseaAvgPool(torch.nn.Module):
    ops_before_pass = {
        "executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 3,
    }
    ops_after_pass = {"executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 1}

    def __init__(self):
        super().__init__()
        self.avg = torch.nn.AvgPool2d(1)

    def forward(self, x):
        return self.avg(x) + self.avg(x) + self.avg(x)


class FuseAvgPoolChain(torch.nn.Module):
    ops_before_pass = {
        "executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 6,
    }
    ops_after_pass = {"executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 2}

    def __init__(self):
        super().__init__()
        self.avg = torch.nn.AvgPool2d(1)

    def forward(self, x):
        first = self.avg(self.avg(x))
        second = self.avg(self.avg(x))
        third = self.avg(self.avg(x))
        return first + second + third


modules = {
    "fuse_avg_pool": FuseaAvgPool(),
    "fuse_avg_pool_chain": FuseAvgPoolChain(),
}


@common.parametrize("module", modules)
def test_fuse_duplicate_ops_FP(module: torch.nn.Module):
    pipeline = PassPipeline[input_t](
        module=module,
        test_data=(torch.ones(1, 1, 1, 1),),
        quantize=False,
        ops_before_pass=module.ops_before_pass,
        ops_after_pass=module.ops_after_pass,
        pass_list=[
            FuseDuplicateUsersPass,
        ],
    )
    pipeline.run()
