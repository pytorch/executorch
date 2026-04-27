# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import executorch.backends.arm.tosa.dialect  # noqa: F401
import pytest
import torch
import tosa_serializer as ts
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.to_tosa_memory_format_pass import (
    ToTosaMemoryFormatPass,
)
from executorch.backends.arm.operators.node_visitor import get_node_visitors
from executorch.backends.arm.process_node import process_call_function
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.backends.test.graph_builder import GraphBuilder
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class _EmitShapePass(ArmPass):
    @property
    def _passes_required_after(self) -> Set[Type[ExportPass]]:
        return set()

    def call_operator(self, op, args, kwargs, meta, updated: bool | None = False):
        # Inject a CONST_SHAPE once, then proceed normally.
        print(f"op: {op}")
        if op == exir_ops.edge.aten.add.Tensor:
            print("Injecting CONST_SHAPE")
            shape = super().call_shape_operator(
                exir_ops.backend.tosa.CONST_SHAPE.default,
                ([1, 3],),
                {},
                meta,
                True,
            )
            return shape
        else:
            return super().call_operator(op, args, kwargs, meta, updated)


def test_const_shape_injects_meta():
    class M(torch.nn.Module):
        def forward(self, x):
            return x + 1

    exported = torch.export.export(M(), (torch.randn(1),))
    edge = to_edge(exported).transform([_EmitShapePass()])

    gm = edge.exported_program().graph_module

    const_shape_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == exir_ops.backend.tosa.CONST_SHAPE.default
    ]

    assert const_shape_nodes
    for n in const_shape_nodes:
        assert n.meta[TosaSpecialDtype.meta_key()] == TosaSpecialDtype.SHAPE


def _graph_module_with_unused_const_shape():
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.1+FP+shape")):
        builder = GraphBuilder()
        builder.call_operator(exir_ops.backend.tosa.CONST_SHAPE.default, ([1],))
        live_const = builder.call_operator(
            exir_ops.backend.tosa.CONST_SHAPE.default, ([3],)
        )
        builder.output([live_const])
        graph_module = ExportPass().call(builder.get_graph_module()).graph_module
        for node in graph_module.graph.nodes:
            if node.op == "call_function":
                node.meta[TosaSpecialDtype.meta_key()] = TosaSpecialDtype.SHAPE
        return graph_module


def _propagate_shape_dim_orders_from_users(graph_module: torch.fx.GraphModule) -> None:
    output_node = next(node for node in graph_module.graph.nodes if node.op == "output")
    output_node.meta["tosa_dim_order"] = (0,)
    dummy_exported = torch.export.export(torch.nn.Identity(), (torch.randn(1),))
    tosa_memory_format_pass = ToTosaMemoryFormatPass(dummy_exported)
    tosa_memory_format_pass._propagate_dim_order_to_shape_args(output_node)


def _serialize_graph_module_to_tosa(graph_module: torch.fx.GraphModule):
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.1+FP+shape")
    node_visitors = get_node_visitors(None, tosa_spec)
    tosa_graph = ts.TosaSerializer(
        "",
        targetMajor=tosa_spec.version.major,
        targetMinor=tosa_spec.version.minor,
        targetPatch=tosa_spec.version.micro,
        targetDraft=True,
    )

    for node in graph_module.graph.nodes:
        if node.op == "call_function":
            process_call_function(node, tosa_graph, node_visitors, tosa_spec)

    return tosa_graph


def test_unused_shape_ops_miss_tosa_dim_order_and_must_be_removed_before_tosa_serialization():
    graph_module = _graph_module_with_unused_const_shape()
    _propagate_shape_dim_orders_from_users(graph_module)

    const_shape_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function"
        and node.target == exir_ops.backend.tosa.CONST_SHAPE.default
    ]
    dead_const_shape, live_const_shape = const_shape_nodes

    assert dead_const_shape.users == {}
    assert "tosa_dim_order" not in dead_const_shape.meta
    assert live_const_shape.meta["tosa_dim_order"] == (0,)

    with pytest.raises(KeyError, match="tosa_dim_order"):
        _serialize_graph_module_to_tosa(graph_module)

    graph_module.graph.eliminate_dead_code()
    graph_module.recompile()

    remaining_const_shape = next(
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function"
        and node.target == exir_ops.backend.tosa.CONST_SHAPE.default
    )
    assert remaining_const_shape.meta["tosa_dim_order"] == (0,)
    assert _serialize_graph_module_to_tosa(graph_module)
