# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, List

import torch

import tosa_serializer as ts

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_same_dtype,
    validate_valid_dtype,
)
from executorch.backends.arm.tosa.mapping import TosaArg


@register_node_visitor
class RshiftVisitor(NodeVisitor):
    target = "aten.bitwise_right_shift.Tensor"

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        validate_num_inputs(self.target, inputs, 2)
        validate_same_dtype(self.target, [*inputs, output], ts)
        validate_valid_dtype(
            self.target,
            [*inputs, output],
            [ts.DType.INT8, ts.DType.INT16, ts.DType.INT32],
            self.tosa_spec,
        )

        attr = ts.TosaSerializerAttribute()
        round = False
        if self.tosa_spec.is_U55_subset:
            # U55 only supports INT32 and round == True
            # TODO MLETORCH-525 Emulate round == False with different decomposition
            round = True
        attr.ArithmeticRightShiftAttribute(round=round)

        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.ARITHMETIC_RIGHT_SHIFT,
            [inputs[0].name, inputs[1].name],
            [output.name],
            attr,
        )
