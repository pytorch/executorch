# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import List

import serializer.tosa_serializer as ts  # type: ignore
import torch.fx
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)

from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_specification import TosaSpecification
from serializer.tosa_serializer import TosaOp


def unary_operator_factory(unary_target: str, tosa_op):
    "Creates and registers NodeVisitors for operations that have one input and map directly into a TOSA op."

    class UnaryOperator_080_MI(NodeVisitor):
        target = unary_target

        tosa_specs = [TosaSpecification.create_from_string("TOSA-0.80+MI")]

        def __init__(self, *args):
            super().__init__(*args)

        def define_node(
            self,
            node: torch.fx.Node,
            tosa_graph: ts.TosaSerializer,
            inputs: List[TosaArg],
            output: TosaArg,
        ) -> None:

            if not (inputs[0].dtype == output.dtype):
                raise ValueError(
                    "All inputs and output need same dtype."
                    f"Got {inputs[0].dtype=}, {output.dtype=}"
                )

            if not (inputs[0].dtype == ts.DType.FP32):
                raise ValueError(
                    "All inputs need to be FP32." f"Got {inputs[0].dtype=}"
                )

            # MI lowering
            tosa_graph.addOperator(tosa_op, [inputs[0].name], [output.name])

    register_node_visitor(UnaryOperator_080_MI)


unary_operator_factory("aten.floor.default", TosaOp.Op().FLOOR)
