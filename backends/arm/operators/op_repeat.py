# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import serializer.tosa_serializer as ts
import torch
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_utils import tosa_shape
from serializer.tosa_serializer import TosaOp


@register_node_visitor
class RepeatVisitor(NodeVisitor):
    target = "aten.repeat.default"

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: list[TosaArg],
        output: TosaArg,
        is_quant_node: bool,
    ) -> None:

        item_name = inputs[0].name
        shape = inputs[0].shape
        rank = len(shape)
        multiples = inputs[1].special
        new_rank = len(multiples)

        assert new_rank >= rank

        # TILE only supports rank(in) == rank(out). To add more dims, we need a reshape first.
        if new_rank > rank:
            # Add length 1 dimensions to shape to match multiples
            num_new_dims = new_rank - rank
            expanded_shape = tuple(
                1 if i < num_new_dims else shape[i - num_new_dims]
                for i in range(new_rank)
            )
            expanded_shape = tosa_shape(expanded_shape, output.dim_order)
            dtype = (
                ts.dtype_str_to_val("INT8")
                if is_quant_node
                else ts.dtype_str_to_val("FP32")
            )

            rescale_out = tosa_graph.addIntermediate(expanded_shape, dtype)
            rescale_attr = ts.TosaSerializerAttribute()
            rescale_attr.ReshapeAttribute(expanded_shape)
            tosa_graph.addOperator(
                TosaOp.Op().RESHAPE, [item_name], [rescale_out.name], rescale_attr
            )
            item_name = rescale_out.name

        attr = ts.TosaSerializerAttribute()
        attr.TileAttribute(tosa_shape(multiples, output.dim_order))
        tosa_graph.addOperator(TosaOp.Op().TILE, [item_name], [output.name], attr)
