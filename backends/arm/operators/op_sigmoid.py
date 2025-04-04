# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import List

import serializer.tosa_serializer as ts  # type: ignore
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_specification import TosaSpecification

from serializer.tosa_serializer import TosaOp
from torch.fx import Node


@register_node_visitor
class SigmoidVisitor_080_MI(NodeVisitor):
    target = "aten.sigmoid.default"

    # BI case should be handled by op_table
    tosa_specs = [TosaSpecification.create_from_string("TOSA-0.80+MI")]

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:

        if len(node.all_input_nodes) != 1:
            raise ValueError(
                f"Expected 1 input for {self.target}, got {len(node.all_input_nodes)}"
            )
        if inputs[0].dtype != ts.DType.FP32 or output.dtype != ts.DType.FP32:
            raise ValueError(
                f"Input and output for {self.target} need to be FP32, got input_dtype: "
                f"{inputs[0].dtype} and output_dtype: {output.dtype}"
            )

        tosa_graph.addOperator(TosaOp.Op().SIGMOID, [inputs[0].name], [output.name])
