# Copyright 2023 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

import serializer.tosa_serializer as ts
import torch
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_utils import is_permute_node_before_addmm
from serializer.tosa_serializer import TosaOp


@register_node_visitor
class PermuteVisitor(NodeVisitor):
    target = "aten.permute_copy.default"

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
        is_quant_node: bool,
    ) -> None:
        if is_permute_node_before_addmm(node):
            ## Simply add an identityOp
            tosa_graph.addOperator(
                TosaOp.Op().IDENTITY, [inputs[0].name], [output.name]
            )
            return

        attr = ts.TosaSerializerAttribute()
        attr.TransposeAttribute(inputs[1].special)
        tosa_graph.addOperator(
            TosaOp.Op().TRANSPOSE, [inputs[0].name], [output.name], attr
        )
