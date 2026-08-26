# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List

import tosa_serializer as ts

from executorch.backends.arm.operators.node_visitor import register_node_visitor
from executorch.backends.arm.operators.simple_node_visitor import (
    SimpleNodeVisitor,
    SimpleNodeVisitorConfig,
)


def binary_operator_factory(
    target: str,
    tosa_op,
    attr_method: str,
    valid_dtypes: List[Any],
):
    operator_target = target

    class BinaryOperator(SimpleNodeVisitor):
        target = operator_target

        @classmethod
        def get_config(cls) -> SimpleNodeVisitorConfig:
            return SimpleNodeVisitorConfig(
                tosa_op=tosa_op,
                attr_method=attr_method,
                num_inputs=2,
                input_dtypes=valid_dtypes,
            )

    register_node_visitor(BinaryOperator)


binary_operator_factory(
    "tosa.BITWISE_AND.default",
    ts.Op.BITWISE_AND,
    "BitwiseAndAttribute",
    [ts.DType.INT8, ts.DType.INT16, ts.DType.INT32],
)
binary_operator_factory(
    "tosa.BITWISE_OR.default",
    ts.Op.BITWISE_OR,
    "BitwiseOrAttribute",
    [ts.DType.INT8, ts.DType.INT16, ts.DType.INT32],
)
binary_operator_factory(
    "tosa.BITWISE_XOR.default",
    ts.Op.BITWISE_XOR,
    "BitwiseXorAttribute",
    [ts.DType.INT8, ts.DType.INT16, ts.DType.INT32],
)
binary_operator_factory(
    "tosa.INTDIV.default",
    ts.Op.INTDIV,
    "IntDivAttribute",
    [ts.DType.INT32],
)
binary_operator_factory(
    "tosa.LOGICAL_AND.default",
    ts.Op.LOGICAL_AND,
    "LogicalAndAttribute",
    [ts.DType.BOOL],
)
binary_operator_factory(
    "tosa.LOGICAL_LEFT_SHIFT.default",
    ts.Op.LOGICAL_LEFT_SHIFT,
    "LogicalLeftShiftAttribute",
    [ts.DType.INT8, ts.DType.INT16, ts.DType.INT32],
)
binary_operator_factory(
    "tosa.LOGICAL_OR.default",
    ts.Op.LOGICAL_OR,
    "LogicalOrAttribute",
    [ts.DType.BOOL],
)
binary_operator_factory(
    "tosa.LOGICAL_RIGHT_SHIFT.default",
    ts.Op.LOGICAL_RIGHT_SHIFT,
    "LogicalRightShiftAttribute",
    [ts.DType.INT8, ts.DType.INT16, ts.DType.INT32],
)
binary_operator_factory(
    "tosa.LOGICAL_XOR.default",
    ts.Op.LOGICAL_XOR,
    "LogicalXorAttribute",
    [ts.DType.BOOL],
)
