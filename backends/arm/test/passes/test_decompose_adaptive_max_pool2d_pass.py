# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.arm.tosa.dialect  # noqa: F401
import torch
from executorch.backends.arm._passes.decompose_adaptive_max_pool2d_pass import (
    DecomposeAdaptiveMaxPool2dPass,
)
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.backends.test.graph_builder import GraphBuilder
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.fx import Node
from torch.fx.passes.infra.pass_base import PassResult


def _graph_module_with_irregular_adaptive_max_pool2d():
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.1+FP+shape")):
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 3, 8, 8))
        # Seed the graph with a representable adaptive pool so fake-op validation
        # can materialize the node; the test mutates it to an irregular case below.
        pool = builder.call_operator(
            exir_ops.backend.tosa.MAX_POOL2D_ADAPTIVE.default,
            (x, [2, 2], [1, 2], [0, 0, 0, 0]),
        )
        builder.output([pool])
        graph_module = ExportPass().call(builder.get_graph_module()).graph_module

    adaptive_node = next(
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function"
        and node.target == exir_ops.backend.tosa.MAX_POOL2D_ADAPTIVE.default
    )
    adaptive_node.args = (adaptive_node.args[0], [3, 3], [2, 2], [0, 0, 0, 0])
    graph_module.recompile()
    return graph_module


def _run_decompose_pass(graph_module):
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.1+FP+shape")):
        result = DecomposeAdaptiveMaxPool2dPass()(graph_module)
    if isinstance(result, PassResult):
        graph_module = result.graph_module
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
    return graph_module


def test_decompose_adaptive_max_pool2d_rewrites_irregular_tosa_op():
    graph_module = _run_decompose_pass(
        _graph_module_with_irregular_adaptive_max_pool2d()
    )

    slice_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function"
        and node.target == exir_ops.backend.tosa.SLICE.default
    ]
    adaptive_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function"
        and node.target == exir_ops.backend.tosa.MAX_POOL2D_ADAPTIVE.default
    ]

    assert len(slice_nodes) == 9
    assert len(adaptive_nodes) == 9

    for node in adaptive_nodes:
        for arg in node.args[1:4]:
            assert isinstance(arg, Node)
            assert arg.target == exir_ops.backend.tosa.CONST_SHAPE.default
