# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Any, List

import executorch.backends.arm.tosa_quant_utils as tqutils

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_specification import TosaSpecification

from torch.fx import Node


@register_node_visitor
class EqualVisitor_0_80(NodeVisitor):
    target = "aten.eq.Tensor"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80+BI"),
        TosaSpecification.create_from_string("TOSA-0.80+MI"),
    ]

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:

        import tosa_tools.v0_80.serializer.tosa_serializer as ts  # type: ignore

        if inputs[0].dtype != inputs[1].dtype:
            raise TypeError(
                "All inputs need to have the same data type for operator EQ but got "
                f"{inputs[0].dtype=}, {inputs[1].dtype=}"
            )

        input_nodes = inputs
        # Handle quantization
        if inputs[0].dtype == ts.DType.INT8:
            # Rescale inputs to 32 bit
            rescaled_inputs, _ = tqutils.insert_rescale_ops_to_int32(
                tosa_graph, inputs, node
            )

            # Update IO
            input_nodes = rescaled_inputs

        # Do the equal comparison
        tosa_graph.addOperator(
            ts.TosaOp.Op().EQUAL,
            [input_nodes[0].name, input_nodes[1].name],
            output.name,
            None,
        )


@register_node_visitor
class EqualVisitor(NodeVisitor):
    target = "aten.eq.Tensor"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
    ]

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:

        import serializer.tosa_serializer as ts  # type: ignore

        if inputs[0].dtype != inputs[1].dtype:
            raise TypeError(
                "All inputs need to have the same data type for operator EQ but got "
                f"{inputs[0].dtype=}, {inputs[1].dtype=}"
            )

        input_nodes = inputs
        # Handle quantization
        if inputs[0].dtype == ts.DType.INT8:
            # Rescale inputs to 32 bit
            rescaled_inputs, _ = tqutils.insert_rescale_ops_to_int32(
                tosa_graph, inputs, node, self.tosa_specs
            )

            # Update IO
            input_nodes = rescaled_inputs

        # Do the equal comparison
        tosa_graph.addOperator(
            ts.TosaOp.Op().EQUAL,
            [input_nodes[0].name, input_nodes[1].name],
            output.name,
            None,
        )
