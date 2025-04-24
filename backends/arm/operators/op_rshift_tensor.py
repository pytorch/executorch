# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

import torch

import tosa_tools.v0_80.serializer.tosa_serializer as ts
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_specification import Tosa_0_80


@register_node_visitor
class RshiftVisitor(NodeVisitor):
    target = "aten.bitwise_right_shift.Tensor"

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:

        attr = ts.TosaSerializerAttribute()
        round = False
        if isinstance(self.tosa_spec, Tosa_0_80) and self.tosa_spec.is_U55_subset:
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
