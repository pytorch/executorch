# Copyright 2026 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import Any, cast, List

import torch

import tosa_serializer as ts  # type: ignore

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa.mapping import TosaArg


@register_node_visitor
class TosaConstShapeVisitor(NodeVisitor):
    target = "tosa.CONST_SHAPE.default"

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        shape_input = inputs[0].special
        tosa_graph = cast(ts.TosaSerializer, tosa_graph)
        tosa_graph.addConst(
            shape_input,
            dtype=ts.DType.SHAPE,
            vals=node.meta["val"],
            name=output.name,
        )
