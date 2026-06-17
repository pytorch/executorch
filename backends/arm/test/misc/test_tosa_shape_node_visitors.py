# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from typing import Any, cast

import pytest
import tosa_serializer as ts  # type: ignore
from executorch.backends.arm.operators.node_visitor import (
    get_node_visitors,
    NodeVisitor,
)
from executorch.backends.arm.tosa.mapping import TosaArg
from executorch.backends.arm.tosa.specification import TosaSpecification
from torch.fx import Node
from tosa.TosaGraph import TosaGraph  # type: ignore[import-not-found, import-untyped]


SHAPE_VISITOR_TARGETS = {
    "tosa.ADD_SHAPE.default",
    "tosa.ASSERT_EQUAL_SHAPE.default",
    "tosa.CONCAT_SHAPE.default",
    "tosa.CONST_SHAPE.default",
    "tosa.DIM.default",
    "tosa.DIV_CEIL_SHAPE.default",
    "tosa.DIV_FLOOR_SHAPE.default",
    "tosa.EXP2_SHAPE.default",
    "tosa.LOG2_CEIL_SHAPE.default",
    "tosa.LOG2_FLOOR_SHAPE.default",
    "tosa.MAX_SHAPE.default",
    "tosa.MIN_SHAPE.default",
    "tosa.MOD_SHAPE.default",
    "tosa.MUL_SHAPE.default",
    "tosa.SLICE_SHAPE.default",
    "tosa.SUB_SHAPE.default",
}

SHAPE_OP_TEST_CASES = [
    ("tosa.ADD_SHAPE.default", ts.Op.ADD_SHAPE, ["lhs", "rhs"], {}),
    (
        "tosa.ASSERT_EQUAL_SHAPE.default",
        ts.Op.ASSERT_EQUAL_SHAPE,
        ["lhs", "rhs"],
        {"allow_broadcast": True},
    ),
    (
        "tosa.ASSERT_EQUAL_SHAPE.default",
        ts.Op.ASSERT_EQUAL_SHAPE,
        ["lhs", "rhs"],
        {"allow_broadcast": False},
    ),
    ("tosa.DIV_CEIL_SHAPE.default", ts.Op.DIV_CEIL_SHAPE, ["lhs", "rhs"], {}),
    ("tosa.DIV_FLOOR_SHAPE.default", ts.Op.DIV_FLOOR_SHAPE, ["lhs", "rhs"], {}),
    ("tosa.EXP2_SHAPE.default", ts.Op.EXP2_SHAPE, ["input"], {}),
    ("tosa.LOG2_CEIL_SHAPE.default", ts.Op.LOG2_CEIL_SHAPE, ["input"], {}),
    ("tosa.LOG2_FLOOR_SHAPE.default", ts.Op.LOG2_FLOOR_SHAPE, ["input"], {}),
    ("tosa.MAX_SHAPE.default", ts.Op.MAX_SHAPE, ["lhs", "rhs"], {}),
    ("tosa.MIN_SHAPE.default", ts.Op.MIN_SHAPE, ["lhs", "rhs"], {}),
    ("tosa.MOD_SHAPE.default", ts.Op.MOD_SHAPE, ["lhs", "rhs"], {}),
    ("tosa.MUL_SHAPE.default", ts.Op.MUL_SHAPE, ["lhs", "rhs"], {}),
    ("tosa.SLICE_SHAPE.default", ts.Op.SLICE_SHAPE, ["input", "start", "size"], {}),
    ("tosa.SUB_SHAPE.default", ts.Op.SUB_SHAPE, ["lhs", "rhs"], {}),
]


def _shape_spec() -> TosaSpecification:
    return TosaSpecification.create_from_string("TOSA-1.1+FP+shape")


def _serializer() -> ts.TosaSerializer:
    return ts.TosaSerializer(
        "",
        targetMajor=1,
        targetMinor=1,
        targetPatch=0,
        targetDraft=True,
    )


def _serialized_op_codes(tosa_graph: ts.TosaSerializer) -> list[ts.Op]:
    graph = TosaGraph.TosaGraph.GetRootAsTosaGraph(tosa_graph.serialize(), 0)
    block = graph.Regions(0).Blocks(0)
    return [block.Operators(index).Op() for index in range(block.OperatorsLength())]


def _define_node(
    visitor: NodeVisitor,
    node: Any,
    tosa_graph: ts.TosaSerializer,
    inputs: list[Any],
    output: Any,
) -> None:
    visitor.define_node(
        cast(Node, node),
        tosa_graph,
        cast(list[TosaArg], inputs),
        cast(TosaArg, output),
    )


def test_all_tosa_shape_ops_have_node_visitors() -> None:
    visitors = get_node_visitors(_shape_spec())

    assert SHAPE_VISITOR_TARGETS <= visitors.keys()


@pytest.mark.parametrize(
    "target, expected_op, input_names, kwargs",
    SHAPE_OP_TEST_CASES,
)
def test_shape_node_visitors_serialize_operator(
    target: str,
    expected_op: ts.Op,
    input_names: list[str],
    kwargs: dict[str, object],
) -> None:
    visitor = get_node_visitors(_shape_spec())[target]
    tosa_graph = _serializer()
    for input_name in input_names:
        tosa_graph.currRegion.currBasicBlock.addShape(input_name, 1)

    _define_node(
        visitor,
        SimpleNamespace(name="node", kwargs=kwargs),
        tosa_graph,
        [SimpleNamespace(name=input_name) for input_name in input_names],
        SimpleNamespace(name="output", shape=(1,)),
    )

    assert _serialized_op_codes(tosa_graph) == [expected_op]


def test_concat_shape_node_visitor_serializes_operator() -> None:
    visitor = get_node_visitors(_shape_spec())["tosa.CONCAT_SHAPE.default"]
    tosa_graph = _serializer()
    input_shapes = [SimpleNamespace(name="shape0"), SimpleNamespace(name="shape1")]
    for shape in input_shapes:
        tosa_graph.currRegion.currBasicBlock.addShape(shape.name, 1)

    _define_node(
        visitor,
        SimpleNamespace(name="node", kwargs={}),
        tosa_graph,
        [SimpleNamespace(special=input_shapes)],
        SimpleNamespace(name="output", shape=(2,)),
    )

    assert _serialized_op_codes(tosa_graph) == [ts.Op.CONCAT_SHAPE]


def test_const_shape_node_visitor_serializes_const_operator() -> None:
    visitor = get_node_visitors(_shape_spec())["tosa.CONST_SHAPE.default"]
    tosa_graph = _serializer()

    _define_node(
        visitor,
        SimpleNamespace(name="node", meta={"val": [2, 3]}, kwargs={}),
        tosa_graph,
        [SimpleNamespace(special=[2, 3])],
        SimpleNamespace(name="output", shape=(2,)),
    )

    assert _serialized_op_codes(tosa_graph) == [ts.Op.CONST_SHAPE]


def test_dim_shape_node_visitor_serializes_operator() -> None:
    visitor = get_node_visitors(_shape_spec())["tosa.DIM.default"]
    tosa_graph = _serializer()
    tosa_graph.currRegion.currBasicBlock.addTensor("input", [1, 2], ts.DType.FP32)

    _define_node(
        visitor,
        SimpleNamespace(name="node", kwargs={"axis": 1}),
        tosa_graph,
        [SimpleNamespace(name="input")],
        SimpleNamespace(name="output", shape=(1,)),
    )

    assert _serialized_op_codes(tosa_graph) == [ts.Op.DIM]
