# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Any, List

import torch
import torch.fx

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


def binary_operator_factory(bw_target: str, tosa_op):
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
            import serializer.tosa_serializer as ts  # type: ignore  # noqa: F401

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
                    output.tosa_spec,
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
                    output.tosa_spec,
                )

            self._serialize_operator(
                node,
                tosa_graph,
                tosa_op,
                [inputs[0].name, inputs[1].name],
                [output.name],
            )

    register_node_visitor(BinaryOperator)


import serializer.tosa_serializer as ts  # type: ignore

binary_operator_factory("aten.bitwise_and.Tensor", ts.TosaOp.Op().BITWISE_AND)
binary_operator_factory("aten.bitwise_xor.Tensor", ts.TosaOp.Op().BITWISE_XOR)
binary_operator_factory("aten.bitwise_or.Tensor", ts.TosaOp.Op().BITWISE_OR)
binary_operator_factory("aten.logical_and.default", ts.TosaOp.Op().LOGICAL_AND)
binary_operator_factory("aten.logical_xor.default", ts.TosaOp.Op().LOGICAL_XOR)
binary_operator_factory("aten.logical_or.default", ts.TosaOp.Op().LOGICAL_OR)
binary_operator_factory(
    "aten.bitwise_left_shift.Tensor", ts.TosaOp.Op().LOGICAL_LEFT_SHIFT
)
