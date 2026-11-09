# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm._passes import DeduplicateConstShapesPass
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Graph, GraphModule


def _const_shape(graph: Graph, name: str, values: list[int]):
    node = graph.call_function(
        exir_ops.backend.tosa.CONST_SHAPE.default,
        (values,),
    )
    node.name = name
    node.meta["val"] = values
    return node


def test_deduplicate_identical_const_shapes():
    graph = Graph()
    first = _const_shape(graph, "first", [2, 3])
    second = _const_shape(graph, "second", [2, 3])
    graph.output((first, second))
    graph_module = GraphModule(torch.nn.Module(), graph)

    result = DeduplicateConstShapesPass()(graph_module)

    assert result is not None
    assert result.modified
    const_shapes = [
        node
        for node in graph_module.graph.nodes
        if node.target == exir_ops.backend.tosa.CONST_SHAPE.default
    ]
    assert [node.name for node in const_shapes] == ["first"]
    assert graph_module.graph.output_node().args[0] == (first, first)


def test_keep_const_shapes_with_different_values():
    graph = Graph()
    first = _const_shape(graph, "first", [2, 3])
    second = _const_shape(graph, "second", [3, 2])
    graph.output((first, second))
    graph_module = GraphModule(torch.nn.Module(), graph)

    result = DeduplicateConstShapesPass()(graph_module)

    assert result is not None
    assert not result.modified
    const_shapes = [
        node
        for node in graph_module.graph.nodes
        if node.target == exir_ops.backend.tosa.CONST_SHAPE.default
    ]
    assert const_shapes == [first, second]
