# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from typing import Any, cast

import pytest
import tosa_serializer as ts
from executorch.backends.arm.operators.op_tosa_rescale import RescaleVisitor
from executorch.backends.arm.tosa.mapping import TosaArg
from executorch.backends.arm.tosa.specification import TosaSpecification
from torch.fx import Node


class CapturingTosaGraph:
    def __init__(self) -> None:
        self.consts: dict[str, tuple[Any, list[int]]] = {}
        self.intermediate_count = 0
        self.operators: list[tuple[Any, tuple[str, ...], tuple[str, ...]]] = []

    def addConst(self, _shape, dtype, values, name):
        self.consts[name] = (dtype, list(values))
        return SimpleNamespace(name=name)

    def addIntermediate(self, shape, dtype):
        self.intermediate_count += 1
        return SimpleNamespace(
            name=f"intermediate_{self.intermediate_count}",
            shape=shape,
            dtype=dtype,
        )

    def addOperator(self, op, inputs, outputs, attributes=None, location=None):
        self.operators.append((op, tuple(inputs), tuple(outputs)))


def _tensor_arg(name: str, dtype: ts.DType) -> TosaArg:
    return cast(TosaArg, SimpleNamespace(name=name, dtype=dtype, shape=(7,)))


def _rescale_unsigned_output(output_zp: int) -> CapturingTosaGraph:
    visitor = RescaleVisitor(TosaSpecification.create_from_string("TOSA-1.0+INT"))
    graph = CapturingTosaGraph()
    visitor._build_rescale(
        node=cast(Node, SimpleNamespace()),
        tosa_graph=graph,
        scale=[1.0],
        input_node=_tensor_arg("wide_input", ts.DType.INT32),
        output=_tensor_arg("unsigned_output", ts.DType.INT8),
        input_zp=[0],
        output_zp=[output_zp],
        rounding_mode=ts.RoundingMode.SINGLE_ROUND,
        input_unsigned=False,
        output_unsigned=True,
    )
    return graph


def test_wide_to_unsigned_int16_rescale_is_rejected() -> None:
    visitor = RescaleVisitor(TosaSpecification.create_from_string("TOSA-1.0+INT"))
    with pytest.raises(
        ValueError,
        match="Wide-to-unsigned RESCALE legalization only supports INT8 output",
    ):
        visitor._build_rescale(
            node=cast(Node, SimpleNamespace()),
            tosa_graph=CapturingTosaGraph(),
            scale=[1.0],
            input_node=_tensor_arg("wide_input", ts.DType.INT32),
            output=_tensor_arg("unsigned_output", ts.DType.INT16),
            input_zp=[0],
            output_zp=[0],
            rounding_mode=ts.RoundingMode.SINGLE_ROUND,
            input_unsigned=False,
            output_unsigned=True,
        )


def _clip(value: int, low: int, high: int) -> int:
    return min(max(value, low), high)


def _direct_unsigned_rescale(value: int, output_zp: int) -> int:
    return _clip(value + output_zp, 0, 255)


def _split_unsigned_rescale(value: int, output_zp: int) -> int:
    signed = _clip(value + output_zp - 128, -128, 127)
    return _clip(signed - (-128), 0, 255)


def test_wide_to_unsigned_rescale_rebases_to_signed_domain() -> None:
    graph = _rescale_unsigned_output(output_zp=0)

    assert len(graph.operators) == 2
    first_op, first_inputs, first_outputs = graph.operators[0]
    second_op, second_inputs, second_outputs = graph.operators[1]

    assert first_op == ts.Op.RESCALE
    assert second_op == ts.Op.RESCALE
    assert first_inputs[0] == "wide_input"
    assert first_outputs == ("intermediate_1",)
    assert second_inputs[0] == "intermediate_1"
    assert second_outputs == ("unsigned_output",)

    assert graph.consts["intermediate_1_output_zp"] == (ts.DType.INT8, [-128])
    assert graph.consts["unsigned_output_input_zp"] == (ts.DType.INT8, [-128])
    assert graph.consts["unsigned_output_output_zp"] == (ts.DType.INT8, [0])


@pytest.mark.parametrize("output_zp", [0, 17, 255])
def test_wide_to_unsigned_rescale_preserves_unsigned_range(output_zp: int) -> None:
    graph = _rescale_unsigned_output(output_zp=output_zp)

    assert graph.consts["intermediate_1_output_zp"] == (
        ts.DType.INT8,
        [output_zp - 128],
    )
    assert graph.consts["unsigned_output_input_zp"] == (ts.DType.INT8, [-128])
    assert graph.consts["unsigned_output_output_zp"] == (ts.DType.INT8, [0])

    for value in (-1, 0, 1, 126, 127, 128, 200, 254, 255, 256):
        assert _split_unsigned_rescale(value, output_zp) == _direct_unsigned_rescale(
            value, output_zp
        )
