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
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_same_dtype,
    validate_valid_dtype,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_specification import TosaSpecification

from torch.fx import Node


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

        validate_num_inputs(self.target, inputs, 2)
        validate_same_dtype(self.target, inputs, ts)
        validate_valid_dtype(
            self.target,
            inputs,
            [ts.DType.INT8, ts.DType.INT32, ts.DType.FP32],
            output.tosa_spec,
        )
        validate_valid_dtype(self.target, output, ts.DType.BOOL, output.tosa_spec)

        input_nodes = inputs
        # Handle quantization
        if inputs[0].dtype == ts.DType.INT8:
            # Rescale inputs to 32 bit
            rescaled_inputs, _ = tqutils.insert_rescale_ops_to_int32(
                tosa_graph, inputs, node, self.tosa_spec
            )

            # Update IO
            input_nodes = rescaled_inputs

        # Do the equal comparison
        self._serialize_operator(
            node,
            tosa_graph,
            ts.TosaOp.Op().EQUAL,
            [input_nodes[0].name, input_nodes[1].name],
            [output.name],
            None,
        )
