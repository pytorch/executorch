# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
        permute_memory_to_nhwc: bool,
    ) -> None:

        # aten.slice_copy supports slicing in 1d at a time.
        # The arguments are dimension of slicing, start index and end index.
        assert len(inputs) == 4
        input, dim, start, end = inputs
        input_rank = len(input.shape)

        dim = dim.number
        output_shape = input.shape
        if permute_memory_to_nhwc and input_rank == 4:
            NCHW_to_NHWC = [0, 3, 1, 2]
            dim = NCHW_to_NHWC[dim]
            NHWC_to_NCHW = [0, 2, 3, 1]
            output_shape = [output_shape[NHWC_to_NCHW[i]] for i in range(input_rank)]

        end = (output_shape[dim] + end.number) % output_shape[dim]
        size = end - start.number
        assert size > 0
        assert size <= output_shape[dim]

        # Convert aten args to Tosa's start and size attributes.
        attr = ts.TosaSerializerAttribute()
        start_attr = [start.number if i == dim else 0 for i in range(input_rank)]
        size_attr = [size if i == dim else output_shape[i] for i in range(input_rank)]
        attr.SliceAttribute(start_attr, size_attr)

        tosa_graph.addOperator(TosaOp.Op().SLICE, [input.name], [output.name], attr)
