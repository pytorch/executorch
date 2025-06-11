# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Any, List

import torch

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_same_dtype,
)
from executorch.backends.arm.tosa_mapping import TosaArg


@register_node_visitor
class RshiftVisitor_0_80(NodeVisitor):
    target = "aten.bitwise_right_shift.Tensor"

    tosa_specs = NodeVisitor.tosa_specs_0_80

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        import tosa_tools.v0_80.serializer.tosa_serializer as ts  # type: ignore

        validate_num_inputs(self.target, inputs, 2)
        validate_same_dtype(self.target, [*inputs, output], ts)

        attr = ts.TosaSerializerAttribute()
        round = False
        if self.tosa_spec.is_U55_subset:
            # U55 only supports INT32 and round == True
            # TODO MLETORCH-525 Emulate round == False with different decomposition
            round = True
        attr.ArithmeticRightShiftAttribute(round=round)

        tosa_graph.addOperator(
            ts.TosaOp.Op().ARITHMETIC_RIGHT_SHIFT,
            [inputs[0].name, inputs[1].name],
            [output.name],
            attr,
        )


@register_node_visitor
class RshiftVisitor(NodeVisitor):
    target = "aten.bitwise_right_shift.Tensor"

    tosa_specs = NodeVisitor.tosa_specs_1_00

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        import serializer.tosa_serializer as ts

        validate_num_inputs(self.target, inputs, 2)
        validate_same_dtype(self.target, [*inputs, output], ts)

        attr = ts.TosaSerializerAttribute()
        round = False
        if self.tosa_spec.is_U55_subset:
            # U55 only supports INT32 and round == True
            # TODO MLETORCH-525 Emulate round == False with different decomposition
            round = True
        attr.ArithmeticRightShiftAttribute(round=round)

        tosa_graph.addOperator(
            ts.TosaOp.Op().ARITHMETIC_RIGHT_SHIFT,
            [inputs[0].name, inputs[1].name],
            [output.name],
            attr,
        )
