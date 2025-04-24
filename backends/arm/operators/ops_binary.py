# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

import torch
import torch.fx

import tosa_tools.v0_80.serializer.tosa_serializer as ts

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg


def binary_operator_factory(bw_target: str, tosa_op):
    """Creates and registers NodeVisitors for operators that have two inputs and map directly to a TOSA op."""

    class BinaryOperator(NodeVisitor):
        target = bw_target

        def define_node(
            self,
            node: torch.fx.Node,
            tosa_graph: ts.TosaSerializer,
            inputs: List[TosaArg],
            output: TosaArg,
        ) -> None:

            if not (inputs[0].dtype == inputs[1].dtype == output.dtype):
                raise ValueError(
                    "All inputs and outputs need same dtype."
                    f"Got {inputs[0].dtype=}, {inputs[1].dtype=}, {output.dtype=}."
                )

            tosa_graph.addOperator(
                tosa_op, [inputs[0].name, inputs[1].name], [output.name]
            )

    register_node_visitor(BinaryOperator)


binary_operator_factory("aten.bitwise_and.Tensor", ts.TosaOp.Op().BITWISE_AND)
binary_operator_factory("aten.bitwise_xor.Tensor", ts.TosaOp.Op().BITWISE_XOR)
binary_operator_factory("aten.bitwise_or.Tensor", ts.TosaOp.Op().BITWISE_OR)
binary_operator_factory("aten.logical_and.default", ts.TosaOp.Op().LOGICAL_AND)
binary_operator_factory("aten.logical_xor.default", ts.TosaOp.Op().LOGICAL_XOR)
binary_operator_factory("aten.logical_or.default", ts.TosaOp.Op().LOGICAL_OR)
binary_operator_factory(
    "aten.bitwise_left_shift.Tensor", ts.TosaOp.Op().LOGICAL_LEFT_SHIFT
)
