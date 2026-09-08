# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from typing import Any, cast

import pytest
import tosa_serializer as ts
from executorch.backends.arm.operators.op_tosa_reverse import TosaReverseVisitor
from executorch.backends.arm.tosa.mapping import TosaArg
from executorch.backends.arm.tosa.specification import TosaSpecification
from torch.fx import Node


class CapturingTosaGraph:
    def __init__(self) -> None:
        self.operators: list[tuple[Any, tuple[Any, ...], tuple[Any, ...], Any, Any]] = (
            []
        )

    def addOperator(self, op, inputs, outputs, attributes=None, location=None):
        self.operators.append((op, tuple(inputs), tuple(outputs), attributes, location))


def _tensor_arg(name: str, dtype) -> SimpleNamespace:
    return SimpleNamespace(name=name, dtype=dtype)


def test_reverse_visitor_emits_tosa_reverse() -> None:
    visitor = TosaReverseVisitor(TosaSpecification.create_from_string("TOSA-1.1+FP"))
    tosa_graph = CapturingTosaGraph()

    visitor.define_node(
        cast(Node, SimpleNamespace(kwargs={"axis": 1})),
        tosa_graph,
        [cast(TosaArg, _tensor_arg("input", ts.DType.FP32))],
        cast(TosaArg, _tensor_arg("output", ts.DType.FP32)),
    )

    assert len(tosa_graph.operators) == 1
    op, inputs, outputs, _attributes, _location = tosa_graph.operators[0]
    assert op == ts.Op.REVERSE
    assert inputs == ("input",)
    assert outputs == ("output",)


def test_reverse_visitor_rejects_bfloat16_without_extension() -> None:
    visitor = TosaReverseVisitor(TosaSpecification.create_from_string("TOSA-1.1+FP"))
    tosa_graph = CapturingTosaGraph()

    with pytest.raises(ValueError):
        visitor.define_node(
            cast(Node, SimpleNamespace(kwargs={"axis": 0})),
            tosa_graph,
            [cast(TosaArg, _tensor_arg("input", ts.DType.BF16))],
            cast(TosaArg, _tensor_arg("output", ts.DType.BF16)),
        )

    assert not tosa_graph.operators
