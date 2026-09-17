# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import executorch.backends.arm.tosa.dialect  # noqa: F401
import torch
from executorch.backends.arm._passes import FuseConsecutiveSlicesPass, RewriteSlicePass
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir.dialects._ops import ops as exir_ops

input_t = Tuple[torch.Tensor]
TOSA_FP_SPEC = TosaSpecification.create_from_string("TOSA-1.0+FP")


class ConsecutiveSlice(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.ops.aten.slice.Tensor(x, 1, 1, 7, 1)
        return torch.ops.aten.slice.Tensor(y, 2, 2, 5, 1)


def _const_shape(graph: torch.fx.Graph, shape: list[int]) -> torch.fx.Node:
    node = graph.create_node(
        "call_function",
        exir_ops.backend.tosa.CONST_SHAPE.default,
        args=(shape,),
    )
    node.meta = {
        "val": shape,
        TosaSpecialDtype.meta_key(): TosaSpecialDtype.SHAPE,
    }
    return node


def _slice(
    graph: torch.fx.Graph,
    input_node: torch.fx.Node,
    start: list[int],
    size: list[int],
    output_shape: tuple[int, ...],
) -> torch.fx.Node:
    slice_node = graph.create_node(
        "call_function",
        exir_ops.backend.tosa.SLICE.default,
        args=(input_node, _const_shape(graph, start), _const_shape(graph, size)),
    )
    slice_node.meta["val"] = torch.ones(output_shape)
    return slice_node


def _slice_nodes(graph_module: torch.fx.GraphModule) -> list[torch.fx.Node]:
    return [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function"
        and node.target == exir_ops.backend.tosa.SLICE.default
    ]


def _shape_arg_value(node: torch.fx.Node, arg_index: int) -> list[int]:
    shape_node = node.args[arg_index]
    assert isinstance(shape_node, torch.fx.Node)
    assert shape_node.target == exir_ops.backend.tosa.CONST_SHAPE.default
    return list(shape_node.args[0])  # type: ignore[arg-type]


def test_fuse_consecutive_slices_combines_static_starts() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.ones((1, 8, 10))
    first_slice = _slice(graph, x, [0, 1, 0], [1, 6, 10], (1, 6, 10))
    second_slice = _slice(graph, first_slice, [0, 2, 3], [1, 2, 4], (1, 2, 4))
    graph.output(second_slice)
    graph_module = torch.fx.GraphModule({}, graph)

    with TosaLoweringContext(TOSA_FP_SPEC):
        result = FuseConsecutiveSlicesPass().call(graph_module)

    slice_nodes = _slice_nodes(result.graph_module)

    assert result.modified
    assert len(slice_nodes) == 1
    assert slice_nodes[0].args[0] is x
    assert _shape_arg_value(slice_nodes[0], 1) == [0, 3, 3]
    assert _shape_arg_value(slice_nodes[0], 2) == [1, 2, 4]


def test_fuse_consecutive_slices_tosa_FP_pipeline() -> None:
    pipeline = PassPipeline[input_t](
        ConsecutiveSlice(),
        (torch.randn(1, 8, 10),),
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor": 2
        },
        ops_after_pass={"backend__ops_tosa_SLICE_default": 1},
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor"
        ],
        pass_list=[RewriteSlicePass, FuseConsecutiveSlicesPass],
    )
    pipeline.pop_stage(-1)

    pipeline.run()
