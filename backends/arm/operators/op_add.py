# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

import executorch.backends.arm.tosa_quant_utils as tqutils
import executorch.backends.arm.tosa_utils as tutils

import serializer.tosa_serializer as ts
import torch
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_specification import TosaSpecification
from serializer.tosa_serializer import TosaOp
from torch.fx import Node


@register_node_visitor
class AddVisitor_080_BI(NodeVisitor):
    target = "aten.add.Tensor"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80+BI"),
    ]

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
        input_nodes = tutils.get_two_inputs(node)

        if not is_quant_node and not all(
            tensor.meta["val"].dtype in (torch.int8, torch.int32)
            for tensor in input_nodes
        ):
            raise RuntimeError(
                f"Unexpected non quantized {AddVisitor_080_BI.target} node."
            )

        needs_rescale = not (
            all(tensor.meta["val"].dtype == torch.int32 for tensor in input_nodes)
            and node.meta["val"].dtype == torch.int32
        )

        if needs_rescale:
            # Rescale inputs to 32 bit
            rescaled_inputs, scale = tqutils.rescale_nodes_to_int32(
                input_nodes, tosa_graph
            )

            # Prepare add output tensor
            broadcasted_shape = tutils.tosa_shape(output.shape, output.dim_order)
            add_output = tosa_graph.addIntermediate(broadcasted_shape, ts.DType.INT32)
        else:
            add_output = output
            rescaled_inputs = inputs

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

        if needs_rescale:
            # Scale output back to 8 bit
            # pyre-ignore
            tqutils.rescale_node_back_to_int8(node, add_output, scale, tosa_graph)


@register_node_visitor
class AddVisitor_080_MI(AddVisitor_080_BI):
    # inheriting 'target' from BI class

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80+MI"),
    ]

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
            # Call the inherited define_node for handling integers
            super().define_node(node, tosa_graph, inputs, output, is_quant_node)
        else:
            # FP32 Add lowering
            tosa_graph.addOperator(
                TosaOp.Op().ADD,
                [inputs[0].name, inputs[1].name],
                [output.name],
                None,
            )
