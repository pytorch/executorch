# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast, List

import torch

import tosa_serializer as ts

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    supported_data_layout_dtypes,
    validate_num_inputs,
    validate_same_dtype,
    validate_valid_dtype,
)
from executorch.backends.arm.tosa.mapping import TosaArg
from executorch.exir.dialects._ops import ops as exir_ops


@register_node_visitor
class TosaPadVisitor(NodeVisitor):
    target = "tosa.PAD.default"

    tosa_specs = NodeVisitor.tosa_specs

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        supported_dtypes = supported_data_layout_dtypes(self.tosa_spec)

        validate_num_inputs(self.target, inputs, 3)
        validate_same_dtype(self.target, [inputs[0], output])
        validate_valid_dtype(
            self.target,
            [inputs[0], output],
            supported_dtypes,
        )
        value_node = cast(torch.fx.Node, node.args[2])
        pad_value_name = inputs[2].name
        if value_node.target == exir_ops.backend.tosa.CONST.default:
            pad_value = tosa_graph.addConst(
                [1],
                output.dtype,
                value_node.args[0],
                name=node.name + "_padding_value",
            )
            pad_value_name = pad_value.name

        attr = ts.TosaSerializerAttribute()
        attr.PadAttribute()

        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.PAD,
            [
                inputs[0].name,
                inputs[1].name,
                pad_value_name,
            ],
            [output.name],
            attr,
        )
