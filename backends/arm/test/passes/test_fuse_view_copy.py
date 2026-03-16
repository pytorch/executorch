# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline
from executorch.backends.transforms.fuse_view_copy import FuseViewCopyTransform


class FuseSequentialViews(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return x.view((1, 2, 3, 4)).view((2, 3, 4, 1)).view((2, 3, 4))

    data = (torch.randn(2, 3, 1, 4),)
    ops_before_pass = {
        "executorch_exir_dialects_edge__ops_aten_view_copy": 3,
    }
    ops_after_pass = {
        "executorch_exir_dialects_edge__ops_aten_view_copy": 1,
    }


class FuseSequentialWithNoopsViews(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return (
            x.view((1, 2, 3, 4))
            .clone()
            .view((2, 3, 4, 1))
            .to(dtype=torch.int32)
            .view((2, 3, 4))
            .abs()
            .reciprocal()
            .sqrt()
            .view((12, 2))
        )

    data = (torch.randn(2, 3, 1, 4),)
    ops_before_pass = {
        "executorch_exir_dialects_edge__ops_aten_view_copy": 4,
    }
    ops_after_pass = {
        "executorch_exir_dialects_edge__ops_aten_view_copy": 1,
    }


class DontFuseBranchingViews(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        x = x.view((1, 2, 3, 4))
        x1 = x.abs().view((2, 3, 4, 1))
        x2 = x.ceil().view((2, 3, 4, 1))
        return x1 + x2

    data = (torch.randn(2, 3, 1, 4),)
    ops_before_pass = {
        "executorch_exir_dialects_edge__ops_aten_view_copy": 3,
    }
    ops_after_pass = {
        "executorch_exir_dialects_edge__ops_aten_view_copy": 3,
    }


tests = {
    "fuse_sequential_views": FuseSequentialViews(),
    "fuse_sequential_with_noops_views": FuseSequentialWithNoopsViews(),
    "dont_fuse_branching_views": DontFuseBranchingViews(),
}


@common.parametrize("model", tests)
def test_fuse_view_copy_transform_tosa_FP(model):
    pipeline = PassPipeline(
        model,
        model.data,
        quantize=False,
        ops_before_pass=model.ops_before_pass,
        ops_after_pass=model.ops_after_pass,
        pass_list=[FuseViewCopyTransform],
    )
    pipeline.run()
