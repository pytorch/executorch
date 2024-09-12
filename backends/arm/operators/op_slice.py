# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

import serializer.tosa_serializer as ts
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from serializer.tosa_serializer import TosaOp
from torch.fx import Node


@register_node_visitor
class SliceVisitor(NodeVisitor):
    target = "aten.slice_copy.Tensor"

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
        is_quant_node: bool,
    ) -> None:

        # aten.slice_copy supports slicing in 1d at a time.
        # The arguments are dimension of slicing, start index and end index.
        assert len(inputs) == 4
        input_node, dim, start, end = inputs

        # Translate and check parameters in Pytorch dim order.
        shape = input_node.shape
        dim = dim.number
        end = (shape[dim] + end.number) % shape[dim]
        if end == 0:
            end = shape[dim]
        size = end - start.number
        assert size > 0
        assert size <= shape[dim]

        # Convert aten args to Tosa's start and size attributes and in TOSA dim order.
        attr = ts.TosaSerializerAttribute()
        start_attr = [start.number if i == dim else 0 for i in input_node.dim_order]
        size_attr = [size if i == dim else shape[i] for i in input_node.dim_order]
        attr.SliceAttribute(start_attr, size_attr)

        tosa_graph.addOperator(
            TosaOp.Op().SLICE, [input_node.name], [output.name], attr
        )
