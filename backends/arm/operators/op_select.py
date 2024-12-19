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

from executorch.backends.arm.tosa_utils import build_reshape, tosa_shape
from serializer.tosa_serializer import TosaOp
from torch.fx import Node


@register_node_visitor
class SelectVisitor(NodeVisitor):
    target = "aten.select_copy.int"

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

        assert len(inputs) == 3
        input_node, dim, index = inputs
        shape = input_node.shape
        rank = len(shape)

        dim = dim.number % rank if dim.number < 0 else dim.number
        index = index.number % rank if index.number < 0 else index.number

        # For aten.select_copy, the output will be rank[input_shape - 1]
        # For TOSA rank(in) == rank(out).
        # Add an intermediate with the same rank
        expanded_shape = tuple(1 if i == dim else shape[i] for i in range(rank))
        expanded_shape = tosa_shape(expanded_shape, input_node.dim_order)

        output_reshaped = tosa_graph.addIntermediate(
            expanded_shape, ts.DType.INT8 if is_quant_node else output.dtype
        )

        attr_slice = ts.TosaSerializerAttribute()

        start_attr = [index if i == dim else 0 for i in input_node.dim_order]
        size_attr = [
            1 if i == dim else input_node.shape[i] for i in input_node.dim_order
        ]

        attr_slice.SliceAttribute(start_attr, size_attr)

        tosa_graph.addOperator(
            TosaOp.Op().SLICE, [input_node.name], [output_reshaped.name], attr_slice
        )

        # Reshape back to original rank of output.
        build_reshape(tosa_graph, output_reshaped.name, output.shape, output.name)
