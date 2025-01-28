# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import List

import serializer.tosa_serializer as ts  # type: ignore
import torch
import tosa.Op as TosaOp  # type: ignore

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg


@register_node_visitor
class ToCopyVisitor(NodeVisitor):
    """
    Implement the type cast functionality of _to_copy.

    Other features like setting of the memory_format or moving a tensor to a
    different device are not supported.

    Also note that the node should not be quantized.
    """

    target = "aten._to_copy.default"

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        tosa_graph.addOperator(TosaOp.Op().CAST, [inputs[0].name], [output.name])
