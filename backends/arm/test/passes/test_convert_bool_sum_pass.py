# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch

from executorch.backends.arm._passes import ConvertBoolSumPass
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from torch._export.utils import _get_shape_env_from_gm
from torch._subclasses import FakeTensorMode
from torch.fx import Graph, GraphModule


class BoolSum(torch.nn.Module):
    def forward(self, x: torch.Tensor, dim: int, keepdim: bool):
        return x.sum(dim=dim, keepdim=keepdim)


def _run_pass(graph_module: GraphModule, tosa_spec: str) -> GraphModule:
    with TosaLoweringContext(
        TosaSpecification.create_from_string(tosa_spec),
        _get_shape_env_from_gm(graph_module),
    ):
        return ConvertBoolSumPass().call(graph_module).graph_module


def _operator_targets(graph_module: GraphModule) -> list[torch._ops.OpOverload]:
    return [
        node.target for node in graph_module.graph.nodes if node.op == "call_function"
    ]


def test_sum_bool_skips_inexact_accumulator() -> None:
    graph = Graph()
    with FakeTensorMode():
        fake_input = torch.empty(torch.iinfo(torch.int32).max + 1, dtype=torch.bool)
        fake_output = torch.empty((), dtype=torch.int64)
    x = graph.placeholder("x")
    x.meta["val"] = fake_input
    output = graph.call_function(torch.ops.aten.sum.dim_IntList, (x, [0], False))
    output.meta["val"] = fake_output
    graph.output(output)

    result = _run_pass(GraphModule(torch.nn.Module(), graph), "TOSA-1.0+INT")

    assert _operator_targets(result) == [torch.ops.aten.sum.dim_IntList]


@pytest.mark.parametrize(
    "max_dynamic_dim,expect_transform",
    [
        (1024, True),
        (torch.iinfo(torch.int32).max + 1, False),
    ],
)
def test_sum_bool_dynamic_shape(max_dynamic_dim: int, expect_transform: bool) -> None:
    dynamic_dim = torch.export.Dim("dynamic_dim", min=1, max=max_dynamic_dim)
    exported_program = torch.export.export(
        BoolSum(),
        (torch.ones(2, dtype=torch.bool), 0, False),
        dynamic_shapes=({0: dynamic_dim}, None, None),
    )

    result = _run_pass(exported_program.graph_module, "TOSA-1.0+INT")

    if not expect_transform:
        assert _operator_targets(result) == [torch.ops.aten.sum.dim_IntList]
        return

    sum_node = next(
        node
        for node in result.graph.nodes
        if node.target == torch.ops.aten.sum.dim_IntList
    )
    (output,) = result(torch.ones(5, dtype=torch.bool), 0, False)
    assert sum_node.kwargs["dtype"] == torch.int32
    assert output.dtype == torch.int64
    assert output == 5


@pytest.mark.parametrize("tosa_spec", ["TOSA-1.0+FP", "TOSA-1.0+FP+INT"])
def test_sum_bool_skips_float_profiles(tosa_spec: str) -> None:
    exported_program = torch.export.export(
        BoolSum(),
        (torch.ones(2, dtype=torch.bool), 0, False),
    )

    result = _run_pass(exported_program.graph_module, tosa_spec)

    assert _operator_targets(result) == [torch.ops.aten.sum.dim_IntList]
