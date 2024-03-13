# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

import serializer.tosa_serializer as ts
import torch
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.op_common import build_avg_pool_2d_common
from executorch.backends.arm.tosa_mapping import TosaArg


@register_node_visitor
class MeanDimVisitor(NodeVisitor):
    target = "aten.mean.dim"

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
        is_quant_node: bool,
    ) -> None:

        input_tensor = inputs[0]
        dim = node.args[1]
        keep_dim = node.args[2]

        # mean.dim(-1, -2) is the same as avg_pool2d when just computing mean over HW dimensions.
        # Since tosa doesn't have mean.dim operation, lowers it to average pooling instead.
        if dim == [-1, -2]:
            if keep_dim is True:
                # Given the shape format of input is (N, C, H, W)
                kernel_size = [input_tensor.shape[2], input_tensor.shape[3]]
                stride = [1, 1]
                padding = [0, 0, 0, 0]

                build_avg_pool_2d_common(
                    node,
                    tosa_graph,
                    input_tensor,
                    kernel_size,
                    stride,
                    padding,
                    is_quant_node,
                    output,
                )
                return

        raise AssertionError("unsupported")
