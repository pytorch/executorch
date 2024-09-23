# Copyright 2023-2024 Arm Limited and/or its affiliates.
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
from executorch.backends.arm.tosa_utils import build_avg_pool_2d_common


@register_node_visitor
class AvgPool2dVisitor(NodeVisitor):
    target = "aten.avg_pool2d.default"

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
        kernel_size_list = inputs[1].special
        stride_size_list = inputs[2].special
        try:
            pad_size_list = inputs[3].special
        except IndexError:
            pad_size_list = [0, 0, 0, 0]

        build_avg_pool_2d_common(
            node,
            tosa_graph,
            input_tensor,
            kernel_size_list,
            stride_size_list,
            pad_size_list,
            is_quant_node,
            output,
        )
