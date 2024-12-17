# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import List

import numpy as np

import serializer.tosa_serializer as ts
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_utils import tosa_shape
from torch.fx import Node


@register_node_visitor
class FullVisitor(NodeVisitor):
    target = "aten.full.default"

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

        shape = tosa_shape(inputs[0].special, output.dim_order)

        value = inputs[1].number

        if output.dtype == ts.DType.INT8:
            fill_dtype = np.int8
        else:
            fill_dtype = np.float32
        data = np.full(shape, value, dtype=fill_dtype)

        tosa_graph.addConst(shape, output.dtype, data, node.name + "full-const")
        tosa_graph.addOperator(
            ts.TosaOp.Op.IDENTITY, [node.name + "full-const"], [output.name]
        )
