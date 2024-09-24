# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

import executorch.backends.arm.tosa_quant_utils as tqutils
import executorch.backends.arm.tosa_utils as tutils

import serializer.tosa_serializer as ts
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from serializer.tosa_serializer import TosaOp
from torch.fx import Node


@register_node_visitor
class AddVisitor(NodeVisitor):
    target = "aten.add.Tensor"

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
        is_quant_node: bool,
    ) -> None:
        if is_quant_node:
            input_nodes = tutils.get_two_inputs(node)

            # Rescale inputs to 32 bit
            rescaled_inputs, scale = tqutils.rescale_nodes_to_int32(
                input_nodes, tosa_graph
            )

            # Preapre sub output tensor
            broadcasted_shape = tutils.broadcast_shapes(
                rescaled_inputs[0].shape, rescaled_inputs[0].shape
            )
            add_output = tosa_graph.addIntermediate(broadcasted_shape, ts.DType.INT32)

            # Do the INT32 Add
            tosa_graph.addOperator(
                TosaOp.Op().ADD,
                [
                    rescaled_inputs[0].name,
                    rescaled_inputs[1].name,
                ],
                [add_output.name],
                None,
            )

            # Scale output back to 8 bit
            tqutils.rescale_node_back_to_int8(node, add_output, scale, tosa_graph)
        else:
            # FP32 Add lowering
            tosa_graph.addOperator(
                TosaOp.Op().ADD,
                [inputs[0].name, inputs[1].name],
                [output.name],
                None,
            )
