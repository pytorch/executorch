# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import cast

import executorch.backends.arm.tosa.dialect  # noqa: F401

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.symbolic_materialization_helper import (
    SymbolMaterializationHelpers,
)
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.backends.test.graph_builder import GraphBuilder
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import NodeMetadata, ProxyValue
from torch.fx import Node


class _ShapeGraphBuilder(GraphBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[torch.fx.Node] = []

    def call_shape_operator(
        self,
        op,
        args: tuple,
        kwargs: dict,
        meta: NodeMetadata,
        updated: bool = True,
    ) -> ProxyValue:
        shape_meta = copy.copy(meta)
        shape_meta.data = dict(meta.data)
        shape_meta.data[TosaSpecialDtype.meta_key()] = TosaSpecialDtype.SHAPE
        proxy = self.call_operator(op, args, kwargs, shape_meta)
        self.calls.append(proxy.node)
        return proxy


def _shape_proxy(builder: GraphBuilder) -> ProxyValue:
    x = builder.placeholder("x", torch.randn(1, 3))
    return builder.call_operator(
        exir_ops.backend.tosa.DIM.default,
        (x,),
        {"axis": 1},
        NodeMetadata(
            {
                "val": [3],
                TosaSpecialDtype.meta_key(): TosaSpecialDtype.SHAPE,
            }
        ),
    )


def _make_helper() -> tuple[SymbolMaterializationHelpers, _ShapeGraphBuilder]:
    builder = _ShapeGraphBuilder()
    helper = SymbolMaterializationHelpers(cast(ArmPass, builder))
    return helper, builder


def test_materialize_int_emits_and_reuses_const_shape() -> None:
    helper, builder = _make_helper()

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.1+FP+shape")):
        first = helper.materialize_arglist([7], NodeMetadata({}))
        second = helper.materialize_arglist([7], NodeMetadata({}))

    assert first.node is second.node
    assert first.node.target == exir_ops.backend.tosa.CONST_SHAPE.default
    assert first.node.args == ([7],)
    assert len(builder.calls) == 1


def test_materialize_single_shape_proxy_reuses_existing_node() -> None:
    helper, builder = _make_helper()

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.1+FP+shape")):
        proxy = _shape_proxy(builder)
        result = helper.materialize_arglist([proxy], NodeMetadata({}))

    assert result.node is proxy.node
    assert builder.calls == []


def test_materialize_arglist_emits_concat_shape_for_mixed_shape_values() -> None:
    helper, builder = _make_helper()

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.1+FP+shape")):
        proxy = _shape_proxy(builder)
        result = helper.materialize_arglist([proxy, 5], NodeMetadata({}))

    assert result.node.target == exir_ops.backend.tosa.CONCAT_SHAPE.default
    assert [node.target for node in builder.calls] == [
        exir_ops.backend.tosa.CONST_SHAPE.default,
        exir_ops.backend.tosa.CONCAT_SHAPE.default,
    ]
    concat_args = result.node.args[0]
    assert isinstance(concat_args, list)
    assert concat_args[0] is proxy.node
    assert builder.calls[0].args == ([5],)


def test_materialize_arglist_flattens_nested_shape_lists() -> None:
    helper, builder = _make_helper()

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.1+FP+shape")):
        proxy = _shape_proxy(builder)
        result = helper.materialize_arglist([[proxy], [2]], NodeMetadata({}))

    assert result.node.target == exir_ops.backend.tosa.CONCAT_SHAPE.default
    concat_args = result.node.args[0]
    assert isinstance(concat_args, list)
    assert concat_args[0] is proxy.node
    assert builder.calls[0].target == exir_ops.backend.tosa.CONST_SHAPE.default
    assert builder.calls[1].target == exir_ops.backend.tosa.CONCAT_SHAPE.default


def test_materialize_arglist_reuses_cached_const_inside_concat() -> None:
    helper, builder = _make_helper()

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.1+FP+shape")):
        proxy = _shape_proxy(builder)
        cached_const = helper.materialize_arglist([5], NodeMetadata({}))
        result = helper.materialize_arglist([proxy, 5], NodeMetadata({}))

    assert result.node.target == exir_ops.backend.tosa.CONCAT_SHAPE.default
    concat_args = result.node.args[0]
    assert isinstance(concat_args, list)
    assert concat_args == [proxy.node, cached_const.node]
    assert [node.target for node in builder.calls] == [
        exir_ops.backend.tosa.CONST_SHAPE.default,
        exir_ops.backend.tosa.CONCAT_SHAPE.default,
    ]


def test_materialize_arglist_accepts_tuple_shape_values() -> None:
    helper, builder = _make_helper()

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.1+FP+shape")):
        proxy = _shape_proxy(builder)
        result = helper.materialize_arglist(((proxy,), (2, 3)), NodeMetadata({}))

    assert result.node.target == exir_ops.backend.tosa.CONCAT_SHAPE.default
    concat_args = result.node.args[0]
    assert isinstance(concat_args, list)
    assert concat_args[0] is proxy.node
    assert builder.calls[0].args == ([2],)
    assert builder.calls[1].args == ([3],)
    assert builder.calls[2].target == exir_ops.backend.tosa.CONCAT_SHAPE.default


def test_materialize_arglist_propagates_meta_to_emitted_shape_ops() -> None:
    helper, _ = _make_helper()
    meta = NodeMetadata({"val": [2, 3], "debug_handle": 123})

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.1+FP+shape")):
        result = helper.materialize_arglist([2, 3], meta)

    concat_args = result.node.args[0]
    assert isinstance(concat_args, list)
    const_node = concat_args[0]
    assert isinstance(const_node, Node)
    assert const_node.meta["debug_handle"] == 123
    assert result.node.meta["debug_handle"] == 123


def test_materialize_shape_op_materializes_non_dim_args() -> None:
    helper, builder = _make_helper()
    meta = NodeMetadata({"val": [4]})

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.1+FP+shape")):
        proxy = _shape_proxy(builder)
        result = helper.materialize_shape_op(
            exir_ops.backend.tosa.ADD_SHAPE.default,
            (proxy, 1),
            {},
            meta,
        )

    assert result.node.target == exir_ops.backend.tosa.ADD_SHAPE.default
    assert result.node.args[0] is proxy.node
    assert builder.calls[0].target == exir_ops.backend.tosa.CONST_SHAPE.default
    assert result.node.args[1] is builder.calls[0]


def test_materialize_shape_op_reuses_cached_output_shape() -> None:
    helper, builder = _make_helper()
    meta = NodeMetadata({"val": [4]})

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.1+FP+shape")):
        proxy = _shape_proxy(builder)
        first = helper.materialize_shape_op(
            exir_ops.backend.tosa.ADD_SHAPE.default,
            (proxy, 1),
            {},
            meta,
        )
        second = helper.materialize_shape_op(
            exir_ops.backend.tosa.ADD_SHAPE.default,
            (proxy, 1),
            {},
            meta,
        )

    assert first.node is second.node
    assert first.node.target == exir_ops.backend.tosa.ADD_SHAPE.default
    assert len(builder.calls) == 2


def test_materialize_dim_does_not_materialize_axis_arg() -> None:
    helper, builder = _make_helper()
    tensor = builder.placeholder("x", torch.randn(1, 3))
    meta = NodeMetadata({"val": [3]})

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.1+FP+shape")):
        result = helper.materialize_shape_op(
            exir_ops.backend.tosa.DIM.default,
            (tensor,),
            {"axis": 1},
            meta,
        )

    assert result.node.target == exir_ops.backend.tosa.DIM.default
    assert result.node.args == (tensor.node,)
    assert result.node.kwargs == {"axis": 1}
    assert len(builder.calls) == 1
