# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from executorch.backends.arm._passes import ConvertPermuteSingletonToViewPass
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline
from executorch.backends.transforms.fuse_view_copy import FuseViewCopyTransform
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


_VIEW = exir_ops.edge.aten.view_copy.default
_PERMUTE = exir_ops.edge.aten.permute_copy.default


def _count_node(
    graph_module: torch.fx.GraphModule, target: torch.fx.node.Target
) -> int:
    return sum(
        node.op == "call_function" and node.target == target
        for node in graph_module.graph.nodes
    )


class _AssertAfterInitialFusePass(ExportPass):
    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        assert _count_node(graph_module, _VIEW) == 1
        assert _count_node(graph_module, _PERMUTE) == 1
        return PassResult(graph_module, False)


class _AssertAfterPermuteToViewPass(ExportPass):
    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        assert _count_node(graph_module, _VIEW) == 2
        assert _count_node(graph_module, _PERMUTE) == 0
        return PassResult(graph_module, False)


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


class FuseViewsIntroducedByLaterPass(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return x.view((1, 2, 1, 3, 4)).view((2, 1, 3, 4)).permute(0, 2, 3, 1)

    data = (torch.randn(2, 1, 3, 4),)


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


def test_fuse_view_copy_transform_runs_again_after_new_fusable_view_tosa_FP():
    model = FuseViewsIntroducedByLaterPass()
    pipeline = PassPipeline(
        model,
        model.data,
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_view_copy": 2,
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_view_copy": 1,
        },
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default",
        ],
        pass_list=[
            FuseViewCopyTransform,
            _AssertAfterInitialFusePass,
            ConvertPermuteSingletonToViewPass,
            _AssertAfterPermuteToViewPass,
            FuseViewCopyTransform,
        ],
    )
    pipeline.run()
