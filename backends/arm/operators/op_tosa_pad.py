# Copyright 2026 Arm Limited and/or its affiliates.
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
from executorch.backends.arm.tosa.mapping import TosaArg


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
        pad_const = tosa_graph.addConst(
            [1],
            output.dtype,
            [node.kwargs.get("value", 0)],
            name=node.name + "_padding_value",
        )

        attr = ts.TosaSerializerAttribute()
        attr.PadAttribute()

        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.PAD,
            [
                inputs[0].name,
                inputs[1].name,
                pad_const.name,
            ],
            [output.name],
            attr,
        )
