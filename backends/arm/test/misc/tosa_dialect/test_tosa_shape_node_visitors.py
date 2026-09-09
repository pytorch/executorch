# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import tosa_serializer as ts  # type: ignore
from executorch.backends.arm.operators.node_visitor import (
    get_node_visitors,
    NodeVisitor,
)
from executorch.backends.arm.test.runner_utils import TosaReferenceModelDispatch
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
    graph = TosaGraph.GetRootAsTosaGraph(tosa_graph.serialize(), 0)
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


def _serialized_tensor_shapes(tosa_buffer: bytes) -> dict[str, list[int]]:
    graph = TosaGraph.TosaGraph.GetRootAsTosaGraph(tosa_buffer, 0)
    block = graph.Regions(0).Blocks(0)
    return {
        block.Tensors(index)
        .Name()
        .decode(): [
            block.Tensors(index).Shape(dim)
            for dim in range(block.Tensors(index).ShapeLength())
        ]
        for index in range(block.TensorsLength())
    }


def _add_const_shape(
    visitors: dict[str, NodeVisitor],
    tosa_graph: ts.TosaSerializer,
    name: str,
    values: list[int],
) -> str:
    _define_node(
        visitors["tosa.CONST_SHAPE.default"],
        SimpleNamespace(name=name, meta={"val": values}, kwargs={}),
        tosa_graph,
        [SimpleNamespace(special=values)],
        SimpleNamespace(name=name, shape=(len(values),)),
    )
    return name


def _add_const_tensor(
    tosa_graph: ts.TosaSerializer,
    name: str,
    values: list[int],
) -> str:
    tosa_graph.addConst([len(values)], ts.DType.INT32, values, name=name)
    return name


def _add_shape_op(
    visitors: dict[str, NodeVisitor],
    tosa_graph: ts.TosaSerializer,
    target: str,
    name: str,
    input_names: list[str],
    *,
    output_rank: int = 1,
    kwargs: dict[str, object] | None = None,
    concat_inputs: bool = False,
) -> str:
    if concat_inputs:
        inputs = [
            SimpleNamespace(special=[SimpleNamespace(name=n) for n in input_names])
        ]
    else:
        inputs = [SimpleNamespace(name=n) for n in input_names]
    _define_node(
        visitors[target],
        SimpleNamespace(name=name, kwargs=kwargs or {}),
        tosa_graph,
        inputs,
        SimpleNamespace(name=name, shape=(output_rank,)),
    )
    return name


def _add_reshape_from_shape(
    tosa_graph: ts.TosaSerializer,
    input_shapes: dict[str, list[int]],
    expected_outputs: dict[str, list[int]],
    name: str,
    shape_name: str,
    expected_shape: list[int],
) -> None:
    data_name = f"data_{name}"
    output_name = f"out_{name}"
    tosa_graph.addInputTensor(
        ts.TosaSerializerTensor(data_name, [-1], ts.DType.INT32, data=None)
    )
    input_shapes[data_name] = [math.prod(expected_shape)]
    tosa_graph.currRegion.currBasicBlock.addTensor(
        output_name,
        [-1] * len(expected_shape),
        ts.DType.INT32,
    )
    attr = ts.TosaSerializerAttribute()
    attr.ReshapeAttribute()
    tosa_graph.addOperator(
        ts.Op.RESHAPE,
        [data_name, shape_name],
        [output_name],
        attr,
    )
    tosa_graph.addOutputTensor(
        tosa_graph.currRegion.currBasicBlock.tensors[output_name]
    )
    expected_outputs[output_name] = expected_shape


def _run_infer_shapes(
    infer_shapes: str,
    tosa_graph: ts.TosaSerializer,
    input_shapes: dict[str, list[int]],
    tmp_path: Path,
) -> dict[str, list[int]]:
    input_names = list(input_shapes)
    inputs = tuple(
        torch.empty(shape, dtype=torch.int32) for shape in input_shapes.values()
    )
    resolved_buffer = TosaReferenceModelDispatch()._run_infer_shapes(
        tosa_graph.serialize(),
        input_names,
        inputs,
        tmp_path,
        infer_shapes,
    )
    return _serialized_tensor_shapes(resolved_buffer)


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


def test_shape_node_visitors_round_trip_through_infer_shapes(tmp_path: Path) -> None:
    infer_shapes = shutil.which("infer_shapes")
    if infer_shapes is None:
        pytest.skip("infer_shapes binary is not available")

    visitors = get_node_visitors(_shape_spec())
    tosa_graph = _serializer()
    input_shapes: dict[str, list[int]] = {}
    expected_outputs: dict[str, list[int]] = {}

    _add_reshape_from_shape(
        tosa_graph,
        input_shapes,
        expected_outputs,
        "const",
        _add_const_shape(visitors, tosa_graph, "shape_const", [6]),
        [6],
    )

    tosa_graph.addInputTensor(
        ts.TosaSerializerTensor("dim_source", [-1], ts.DType.INT32, data=None)
    )
    input_shapes["dim_source"] = [6]
    _add_reshape_from_shape(
        tosa_graph,
        input_shapes,
        expected_outputs,
        "dim",
        _add_shape_op(
            visitors,
            tosa_graph,
            "tosa.DIM.default",
            "shape_dim",
            ["dim_source"],
            kwargs={"axis": 0},
        ),
        [6],
    )

    _add_reshape_from_shape(
        tosa_graph,
        input_shapes,
        expected_outputs,
        "add",
        _add_shape_op(
            visitors,
            tosa_graph,
            "tosa.ADD_SHAPE.default",
            "shape_add",
            [
                _add_const_shape(visitors, tosa_graph, "add_lhs", [4]),
                _add_const_shape(visitors, tosa_graph, "add_rhs", [2]),
            ],
        ),
        [6],
    )
    _add_reshape_from_shape(
        tosa_graph,
        input_shapes,
        expected_outputs,
        "sub",
        _add_shape_op(
            visitors,
            tosa_graph,
            "tosa.SUB_SHAPE.default",
            "shape_sub",
            [
                _add_const_shape(visitors, tosa_graph, "sub_lhs", [8]),
                _add_const_shape(visitors, tosa_graph, "sub_rhs", [2]),
            ],
        ),
        [6],
    )
    _add_reshape_from_shape(
        tosa_graph,
        input_shapes,
        expected_outputs,
        "mul",
        _add_shape_op(
            visitors,
            tosa_graph,
            "tosa.MUL_SHAPE.default",
            "shape_mul",
            [
                _add_const_shape(visitors, tosa_graph, "mul_lhs", [3]),
                _add_const_shape(visitors, tosa_graph, "mul_rhs", [2]),
            ],
        ),
        [6],
    )
    _add_reshape_from_shape(
        tosa_graph,
        input_shapes,
        expected_outputs,
        "div_ceil",
        _add_shape_op(
            visitors,
            tosa_graph,
            "tosa.DIV_CEIL_SHAPE.default",
            "shape_div_ceil",
            [
                _add_const_shape(visitors, tosa_graph, "div_ceil_lhs", [11]),
                _add_const_shape(visitors, tosa_graph, "div_ceil_rhs", [2]),
            ],
        ),
        [6],
    )
    _add_reshape_from_shape(
        tosa_graph,
        input_shapes,
        expected_outputs,
        "div_floor",
        _add_shape_op(
            visitors,
            tosa_graph,
            "tosa.DIV_FLOOR_SHAPE.default",
            "shape_div_floor",
            [
                _add_const_shape(visitors, tosa_graph, "div_floor_lhs", [13]),
                _add_const_shape(visitors, tosa_graph, "div_floor_rhs", [2]),
            ],
        ),
        [6],
    )
    _add_reshape_from_shape(
        tosa_graph,
        input_shapes,
        expected_outputs,
        "mod",
        _add_shape_op(
            visitors,
            tosa_graph,
            "tosa.MOD_SHAPE.default",
            "shape_mod",
            [
                _add_const_shape(visitors, tosa_graph, "mod_lhs", [20]),
                _add_const_shape(visitors, tosa_graph, "mod_rhs", [7]),
            ],
        ),
        [6],
    )
    _add_reshape_from_shape(
        tosa_graph,
        input_shapes,
        expected_outputs,
        "max",
        _add_shape_op(
            visitors,
            tosa_graph,
            "tosa.MAX_SHAPE.default",
            "shape_max",
            [
                _add_const_shape(visitors, tosa_graph, "max_lhs", [6]),
                _add_const_shape(visitors, tosa_graph, "max_rhs", [2]),
            ],
        ),
        [6],
    )
    _add_reshape_from_shape(
        tosa_graph,
        input_shapes,
        expected_outputs,
        "min",
        _add_shape_op(
            visitors,
            tosa_graph,
            "tosa.MIN_SHAPE.default",
            "shape_min",
            [
                _add_const_shape(visitors, tosa_graph, "min_lhs", [6]),
                _add_const_shape(visitors, tosa_graph, "min_rhs", [8]),
            ],
        ),
        [6],
    )
    _add_reshape_from_shape(
        tosa_graph,
        input_shapes,
        expected_outputs,
        "exp2",
        _add_shape_op(
            visitors,
            tosa_graph,
            "tosa.EXP2_SHAPE.default",
            "shape_exp2",
            [_add_const_shape(visitors, tosa_graph, "exp2_in", [3])],
        ),
        [8],
    )
    _add_reshape_from_shape(
        tosa_graph,
        input_shapes,
        expected_outputs,
        "log2_ceil",
        _add_shape_op(
            visitors,
            tosa_graph,
            "tosa.LOG2_CEIL_SHAPE.default",
            "shape_log2_ceil",
            [_add_const_shape(visitors, tosa_graph, "log2_ceil_in", [9])],
        ),
        [4],
    )
    _add_reshape_from_shape(
        tosa_graph,
        input_shapes,
        expected_outputs,
        "log2_floor",
        _add_shape_op(
            visitors,
            tosa_graph,
            "tosa.LOG2_FLOOR_SHAPE.default",
            "shape_log2_floor",
            [_add_const_shape(visitors, tosa_graph, "log2_floor_in", [15])],
        ),
        [3],
    )
    _add_reshape_from_shape(
        tosa_graph,
        input_shapes,
        expected_outputs,
        "concat",
        _add_shape_op(
            visitors,
            tosa_graph,
            "tosa.CONCAT_SHAPE.default",
            "shape_concat",
            [
                _add_const_shape(visitors, tosa_graph, "concat_lhs", [2]),
                _add_const_shape(visitors, tosa_graph, "concat_rhs", [4]),
            ],
            output_rank=2,
            concat_inputs=True,
        ),
        [2, 4],
    )
    _add_reshape_from_shape(
        tosa_graph,
        input_shapes,
        expected_outputs,
        "slice",
        _add_shape_op(
            visitors,
            tosa_graph,
            "tosa.SLICE_SHAPE.default",
            "shape_slice",
            [
                _add_const_shape(visitors, tosa_graph, "slice_input", [2, 6, 4]),
                _add_const_tensor(tosa_graph, "slice_start", [1]),
                _add_const_tensor(tosa_graph, "slice_size", [1]),
            ],
        ),
        [6],
    )
    _add_shape_op(
        visitors,
        tosa_graph,
        "tosa.ASSERT_EQUAL_SHAPE.default",
        "shape_assert",
        [
            _add_const_shape(visitors, tosa_graph, "assert_lhs", [6]),
            _add_const_shape(visitors, tosa_graph, "assert_rhs", [6]),
        ],
        kwargs={"allow_broadcast": False},
    )

    resolved_shapes = _run_infer_shapes(
        infer_shapes,
        tosa_graph,
        input_shapes,
        tmp_path,
    )

    assert {
        name: resolved_shapes[name] for name in expected_outputs
    } == expected_outputs
