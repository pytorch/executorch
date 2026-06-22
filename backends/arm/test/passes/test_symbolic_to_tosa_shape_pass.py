# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import executorch.backends.arm.tosa.dialect  # noqa: F401
import torch
from executorch.backends.arm._passes.symbolic_to_tosa_shape_pass import (
    SymbolicToTosaShapesPass,
)
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.backends.test.graph_builder import GraphBuilder
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import NodeMetadata, PassResult, ProxyValue


def _run_symbolic_shape_pass(graph_module: torch.fx.GraphModule) -> PassResult:
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.1+FP+shape")):
        result = SymbolicToTosaShapesPass()(graph_module)
    assert result is not None
    return result


def _single_node_with_target(
    graph_module: torch.fx.GraphModule,
    target: Any,
) -> torch.fx.Node:
    return next(node for node in graph_module.graph.nodes if node.target == target)


def _sym_size(
    builder: GraphBuilder,
    x: ProxyValue,
    dim: int,
    value: int,
) -> ProxyValue:
    return builder.call_operator(
        torch.ops.aten.sym_size.int,
        (x, dim),
        meta=NodeMetadata({"val": value}),
    )


def test_symbolic_to_tosa_shapes_rewrites_sym_size_to_dim() -> None:
    builder = GraphBuilder()
    x = builder.placeholder("x", torch.randn(2, 18))
    sym_size = _sym_size(builder, x, 1, 18)
    builder.output([sym_size])

    result = _run_symbolic_shape_pass(builder.get_graph_module())
    graph_module = result.graph_module
    dim_node = _single_node_with_target(graph_module, exir_ops.backend.tosa.DIM.default)

    assert getattr(dim_node.args[0], "target", None) == "x"
    assert dim_node.kwargs == {"axis": 1}
    assert torch.ops.aten.sym_size.int not in {
        node.target for node in graph_module.graph.nodes
    }
    assert result.modified is True
    graph_module.graph.lint()


def test_symbolic_to_tosa_shapes_marks_dim_as_shape_dtype() -> None:
    builder = GraphBuilder()
    x = builder.placeholder("x", torch.randn(2, 18))
    sym_size = _sym_size(builder, x, 0, 2)
    builder.output([sym_size])

    result = _run_symbolic_shape_pass(builder.get_graph_module())
    dim_node = _single_node_with_target(
        result.graph_module, exir_ops.backend.tosa.DIM.default
    )

    assert dim_node.meta[TosaSpecialDtype.meta_key()] == TosaSpecialDtype.SHAPE
    result.graph_module.graph.lint()


def test_symbolic_to_tosa_shapes_leaves_non_sym_size_ops_unchanged() -> None:
    builder = GraphBuilder()
    x = builder.placeholder("x", torch.randn(2, 18))
    add = builder.call_operator(
        torch.ops.aten.add.Tensor,
        (x, x),
        meta=NodeMetadata({"val": torch.empty(2, 18)}),
    )
    builder.output([add])

    result = _run_symbolic_shape_pass(builder.get_graph_module())
    graph_module = result.graph_module

    _single_node_with_target(graph_module, torch.ops.aten.add.Tensor)
    assert exir_ops.backend.tosa.DIM.default not in {
        node.target for node in graph_module.graph.nodes
    }
    graph_module.graph.lint()
