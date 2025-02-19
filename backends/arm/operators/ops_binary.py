# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

import serializer.tosa_serializer as ts
import torch
import torch.fx

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from serializer.tosa_serializer import TosaOp


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


binary_operator_factory("aten.bitwise_and.Tensor", TosaOp.Op().BITWISE_AND)
binary_operator_factory("aten.bitwise_xor.Tensor", TosaOp.Op().BITWISE_XOR)
binary_operator_factory("aten.bitwise_or.Tensor", TosaOp.Op().BITWISE_OR)
