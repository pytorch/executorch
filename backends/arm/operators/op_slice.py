# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

import tosa_tools.v0_80.serializer.tosa_serializer as ts  # type: ignore
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from torch.fx import Node


@register_node_visitor
class SliceVisitor(NodeVisitor):
    target = "aten.slice_copy.Tensor"

    def __init__(self, *args):
        super().__init__(*args)

    def _fixup_start(self, start, shape, dim):
        if start.number < 0:
            return start.number % shape[dim]
        else:
            return start.number

    def _fixup_end(self, end, shape, dim):
        if end.number < 0:
            return end.number % shape[dim]
        else:
            return min(end.number, shape[dim])

    def define_node(
        self,
        node: Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:

        # See slice_copy_support.py
        if not (len(inputs) == 4 or (len(inputs) == 5 and inputs[4].number == 1)):
            raise ValueError("Unsupported combination of inputs")

        # aten.slice_copy supports slicing in 1d at a time.
        # The arguments are the actual input, dimension of slicing, start index, end index and optinal step or stride.
        input_node, dim, start, end = inputs

        # Translate and check parameters in Pytorch dim order.
        shape = input_node.shape
        dim = dim.number

        start_index = self._fixup_start(start, shape, dim)
        end_index = self._fixup_end(end, shape, dim)
        size = end_index - start_index

        assert size > 0
        assert size <= shape[dim]

        # Convert aten args to Tosa's start and size attributes and in TOSA dim order.
        attr = ts.TosaSerializerAttribute()

        start_attr = [
            self._fixup_start(start, shape, dim) if i == dim else 0
            for i in input_node.dim_order
        ]
        size_attr = [size if i == dim else shape[i] for i in input_node.dim_order]
        attr.SliceAttribute(start_attr, size_attr)

        tosa_graph.addOperator(
            ts.TosaOp.Op().SLICE, [input_node.name], [output.name], attr
        )
