# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Callable, List

import torch
import torch.fx

import tosa_serializer as ts

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_same_dtype,
    validate_valid_dtype,
)
from executorch.backends.arm.tosa.mapping import TosaArg


def binary_operator_factory(
    bw_target: str, tosa_op, attr_builder: Callable[[Any], None]
):
    """Creates and registers NodeVisitors for operators that have two inputs and map directly to a TOSA op."""

    class BinaryOperator(NodeVisitor):
        target = bw_target
        tosa_specs = NodeVisitor.tosa_specs

        def define_node(
            self,
            node: torch.fx.Node,
            tosa_graph: Any,
            inputs: List[TosaArg],
            output: TosaArg,
        ) -> None:
            validate_num_inputs(self.target, inputs, 2)
            validate_same_dtype(self.target, [*inputs, output], ts)

            if self.target in [
                "aten.bitwise_and.Tensor",
                "aten.bitwise_xor.Tensor",
                "aten.bitwise_or.Tensor",
                "aten.bitwise_left_shift.Tensor",
            ]:
                validate_valid_dtype(
                    self.target,
                    [*inputs, output],
                    [ts.DType.INT8, ts.DType.INT16, ts.DType.INT32],
                    self.tosa_spec,
                )
            if self.target in [
                "aten.logical_and.default",
                "aten.logical_xor.defaul",
                "aten.logical_or.default",
            ]:
                validate_valid_dtype(
                    self.target,
                    [*inputs, output],
                    [ts.DType.BOOL],
                    self.tosa_spec,
                )
            attr = ts.TosaSerializerAttribute()
            attr_builder(attr)
            self._serialize_operator(
                node,
                tosa_graph,
                tosa_op,
                [inputs[0].name, inputs[1].name],
                [output.name],
                attr,
            )

    register_node_visitor(BinaryOperator)


binary_operator_factory(
    "aten.bitwise_and.Tensor",
    ts.Op.BITWISE_AND,
    lambda attr: attr.BitwiseAndAttribute(),
)
binary_operator_factory(
    "aten.bitwise_xor.Tensor",
    ts.Op.BITWISE_XOR,
    lambda attr: attr.BitwiseXorAttribute(),
)
binary_operator_factory(
    "aten.bitwise_or.Tensor", ts.Op.BITWISE_OR, lambda attr: attr.BitwiseOrAttribute()
)
binary_operator_factory(
    "aten.logical_and.default",
    ts.Op.LOGICAL_AND,
    lambda attr: attr.LogicalAndAttribute(),
)
binary_operator_factory(
    "aten.logical_xor.default",
    ts.Op.LOGICAL_XOR,
    lambda attr: attr.LogicalXorAttribute(),
)
binary_operator_factory(
    "aten.logical_or.default", ts.Op.LOGICAL_OR, lambda attr: attr.LogicalOrAttribute()
)
binary_operator_factory(
    "aten.bitwise_left_shift.Tensor",
    ts.Op.LOGICAL_LEFT_SHIFT,
    lambda attr: attr.LogicalLeftShiftAttribute(),
)
