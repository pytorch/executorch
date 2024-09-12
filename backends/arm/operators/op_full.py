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
from executorch.backends.arm.tosa_quant_utils import get_quant_node_args
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
        if is_quant_node:
            qargs = get_quant_node_args(list(node.users)[0])
            qvalue = np.clip(
                np.round(value / qargs.scale) + qargs.zp, qargs.qmin, qargs.qmax
            )
            dtype = ts.DType.INT8
            data = np.full(shape, qvalue, dtype=np.int8)
        else:
            assert (
                output.dtype == ts.DType.FP32
            ), "'Full' currently only supports FP32 for unquantized models."
            dtype = ts.DType.FP32
            data = np.full(shape, value, dtype=np.float32)

        tosa_graph.addConst(shape, dtype, data, "full-const")
        tosa_graph.addOperator(ts.TosaOp.Op.IDENTITY, ["full-const"], [output.name])
