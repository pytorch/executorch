# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import List

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
class SqueezeVisitor(NodeVisitor):
    target = "aten.squeeze_copy.dims"

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
        is_quant_node: bool,
    ) -> None:
        shape = inputs[0].shape
        rank = len(shape)
        # In some cases, e.g. torch.randn((1, 5, 1, 5)).squeeze(),
        # dims == [0, 1, 2, 3] even though all dims cannot be squeezed.
        # We need to verify that shape[dim] == 1 before squeezing the dim.
        dims = [dim % rank for dim in inputs[1].special if shape[dim] == 1]
        new_shape = [shape[i] for i in range(rank) if i not in dims]
        new_shape = tosa_shape(new_shape, output.dim_order)
        attr = ts.TosaSerializerAttribute()
        attr.ReshapeAttribute(new_shape)
        tosa_graph.addOperator(
            TosaOp.Op().RESHAPE, [inputs[0].name], [output.name], attr
        )
