# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, List

import tosa_serializer as ts

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa.mapping import TosaArg
from torch.fx import Node


@register_node_visitor
class SliceVisitor(NodeVisitor):
    target = "tosa.SLICE.default"

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:

        input_node, starts, sizes = inputs
        attr = ts.TosaSerializerAttribute()
        attr.SliceAttribute()
        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.SLICE,
            [input_node.name, starts.name, sizes.name],
            [output.name],
            attr,
        )
