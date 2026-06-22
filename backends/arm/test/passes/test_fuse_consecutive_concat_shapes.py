# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import executorch.backends.arm.tosa.dialect  # noqa: F401
from executorch.backends.arm._passes.fuse_consecutive_concat_shapes import (
    FuseConsecutiveConcatShapesPass,
)
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.backends.test.graph_builder import GraphBuilder
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_base import PassResult


def _graph_module_with_nested_concat():
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.1+FP+shape")):
        builder = GraphBuilder()
        const_0 = builder.call_operator(
            exir_ops.backend.tosa.CONST_SHAPE.default, ([0],)
        )
        const_1 = builder.call_operator(
            exir_ops.backend.tosa.CONST_SHAPE.default, ([1],)
        )
        const_2 = builder.call_operator(
            exir_ops.backend.tosa.CONST_SHAPE.default, ([2],)
        )
        const_3 = builder.call_operator(
            exir_ops.backend.tosa.CONST_SHAPE.default, ([3],)
        )
        inner = builder.call_operator(
            exir_ops.backend.tosa.CONCAT_SHAPE.default, ([const_1, const_2],)
        )
        outer = builder.call_operator(
            exir_ops.backend.tosa.CONCAT_SHAPE.default, ([const_0, inner, const_3],)
        )
        builder.output([outer])
        return ExportPass().call(builder.get_graph_module()).graph_module


def _graph_module_with_flat_concat():
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.1+FP+shape")):
        builder = GraphBuilder()
        const_0 = builder.call_operator(
            exir_ops.backend.tosa.CONST_SHAPE.default, ([4],)
        )
        const_1 = builder.call_operator(
            exir_ops.backend.tosa.CONST_SHAPE.default, ([5],)
        )
        const_2 = builder.call_operator(
            exir_ops.backend.tosa.CONST_SHAPE.default, ([6],)
        )
        outer = builder.call_operator(
            exir_ops.backend.tosa.CONCAT_SHAPE.default, ([const_0, const_1, const_2],)
        )
        builder.output([outer])
        return ExportPass().call(builder.get_graph_module()).graph_module


def _concat_shape_nodes(graph_module):
    return [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function"
        and node.target == exir_ops.backend.tosa.CONCAT_SHAPE.default
    ]


def _const_shape_values(shape_list_nodes):
    return [node.args[0][0] for node in shape_list_nodes]


def _run_fuse_pass(graph_module: GraphModule):
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.1+FP+shape")):
        result = FuseConsecutiveConcatShapesPass()(graph_module)
    if isinstance(result, PassResult):
        graph_module = result.graph_module
        graph_module.graph.eliminate_dead_code()
    return graph_module


def test_fuse_consecutive_concat_shapes_flattens_nested_concat_inputs():
    graph_module = _graph_module_with_nested_concat()
    graph_module = _run_fuse_pass(graph_module)

    concat_nodes = _concat_shape_nodes(graph_module)
    outer_concat = concat_nodes[-1]
    outer_inputs = outer_concat.args[0]

    assert len(concat_nodes) == 1
    assert _const_shape_values(outer_inputs) == [0, 1, 2, 3]
    assert all(
        node.target == exir_ops.backend.tosa.CONST_SHAPE.default
        for node in outer_inputs
    )


def test_fuse_consecutive_concat_shapes_leaves_flat_concat_unchanged():
    graph_module = _graph_module_with_flat_concat()
    graph_module = _run_fuse_pass(graph_module)

    concat_nodes = _concat_shape_nodes(graph_module)
    outer_inputs = concat_nodes[-1].args[0]

    assert len(concat_nodes) == 1
    assert _const_shape_values(outer_inputs) == [4, 5, 6]
