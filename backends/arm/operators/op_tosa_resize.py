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
)
from executorch.backends.arm.tosa.mapping import TosaArg


@register_node_visitor
class ResizeVisitor(NodeVisitor):
    target = "tosa.RESIZE.default"

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        x, scales, offset, border = inputs
        validate_num_inputs(self.target, inputs, [4])
        if node.kwargs.get("resize_mode") == "bilinear":
            resize_mode = ts.ResizeMode.BILINEAR
        else:
            resize_mode = ts.ResizeMode.NEAREST
        attr = ts.TosaSerializerAttribute()
        attr.ResizeAttribute(resize_mode)

        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.RESIZE,
            [
                x.name,
                scales.name,
                offset.name,
                border.name,
            ],
            [output.name],
            attr,
        )
