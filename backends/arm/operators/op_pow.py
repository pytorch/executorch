# Copyright 2025 Arm Limited and/or its affiliates.
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
class PowVisitor_080_MI(NodeVisitor):
    target = "aten.pow.Tensor_Tensor"

    tosa_specs = [
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

        validate_num_inputs(self.target, inputs, 2)
        validate_same_dtype(self.target, [*inputs, output], ts)

        if inputs[0].dtype not in [ts.DType.FP32, ts.DType.FP16]:
            raise ValueError(
                f"All inputs need to be FP32 or FP16. Got {inputs[0].dtype}"
            )

        tosa_graph.addOperator(
            ts.TosaOp.Op().POW,
            [
                inputs[0].name,
                inputs[1].name,
            ],
            [output.name],
            None,
        )


@register_node_visitor
class PowVisitor(NodeVisitor):
    target = "aten.pow.Tensor_Tensor"

    tosa_specs = [
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
        import serializer.tosa_serializer as ts

        validate_num_inputs(self.target, inputs, 2)
        validate_same_dtype(self.target, [*inputs, output], ts)

        if inputs[0].dtype not in [ts.DType.FP32, ts.DType.FP16]:
            raise ValueError(
                f"All inputs need to be FP32 or FP16. Got {inputs[0].dtype}"
            )

        tosa_graph.addOperator(
            ts.TosaOp.Op().POW,
            [
                inputs[0].name,
                inputs[1].name,
            ],
            [output.name],
            None,
        )
