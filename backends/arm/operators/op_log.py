# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import Any, List

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_same_dtype,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_specification import TosaSpecification
from torch.fx import Node


@register_node_visitor
class LogVisitor_0_80_MI(NodeVisitor):
    target = "aten.log.default"

    # BI case should be handled by op_table
    tosa_specs = [TosaSpecification.create_from_string("TOSA-0.80+MI")]

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

        validate_num_inputs(self.target, inputs, 1)
        validate_same_dtype(self.target, [*inputs, output], ts)

        if inputs[0].dtype != ts.DType.FP32 or output.dtype != ts.DType.FP32:
            raise ValueError(
                f"Input and output for {self.target} need to be FP32, got input_dtype: "
                f"{inputs[0].dtype} and output_dtype: {output.dtype}"
            )

        tosa_graph.addOperator(ts.TosaOp.Op().LOG, [inputs[0].name], [output.name])


@register_node_visitor
class LogVisitor(NodeVisitor):
    target = "aten.log.default"

    # INT case should be handled by op_table
    tosa_specs = [TosaSpecification.create_from_string("TOSA-1.0+FP")]

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        import serializer.tosa_serializer as ts

        validate_num_inputs(self.target, inputs, 1)
        validate_same_dtype(self.target, [*inputs, output], ts)

        if inputs[0].dtype != ts.DType.FP32 or output.dtype != ts.DType.FP32:
            raise ValueError(
                f"Input and output for {self.target} need to be FP32, got input_dtype: "
                f"{inputs[0].dtype} and output_dtype: {output.dtype}"
            )

        tosa_graph.addOperator(ts.TosaOp.Op().LOG, [inputs[0].name], [output.name])
