#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

from typing import cast, List

import torch
from executorch.backends.apple.mps.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.apple.mps.serialization.mps_graph_schema import (
    MPSAvgPool2D,
    MPSGraph,
    MPSMaxPool2DWithIndices,
    MPSNode,
)
from executorch.backends.apple.mps.utils.mps_utils import get_input_node


@register_node_visitor
class MaxPool2DWithIndicesVisitor(NodeVisitor):
    target = "aten.max_pool2d_with_indices.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        n_args = len(node.args)
        if n_args > 6:
            raise AssertionError(
                f"Unexpected number of input parameters for {self.target}"
            )

        input1_id = self.define_tensor(get_input_node(node, 0), mps_graph)

        padding = [0, 0]
        dilation = [1, 1]
        ceil_mode = False
        kernel_size = cast(List[int], node.args[1])
        stride = cast(List[int], node.args[2])
        if n_args >= 4:
            padding = cast(List[int], node.args[3])
        if n_args >= 5:
            dilation = cast(List[int], node.args[4])
        if n_args == 6:
            ceil_mode = cast(bool, node.args[5])
        padding_top = padding[0]
        padding_left = padding[1]
        padding_bottom = padding[0] * stride[0] if ceil_mode else padding[0]
        padding_right = padding[1] * stride[1] if ceil_mode else padding[1]

        output1_id, output2_id = self.define_tensor_list(node, mps_graph)
        mps_graph.mps_nodes.append(
            MPSNode(
                mpsnode_union=MPSMaxPool2DWithIndices(
                    input1_id=input1_id,
                    kernel_height=kernel_size[0],
                    kernel_width=kernel_size[1],
                    stride_height=stride[0],
                    stride_width=stride[1],
                    padding_left=padding_left,
                    padding_right=padding_right,
                    padding_top=padding_top,
                    padding_bottom=padding_bottom,
                    dilation_height=dilation[0],
                    dilation_width=dilation[1],
                    ceil_mode=ceil_mode,
                    output1_id=output1_id,
                    output2_id=output2_id,
                )
            )
        )


@register_node_visitor
class AvgPool2DVisitor(NodeVisitor):
    target = "aten.avg_pool2d.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        n_args = len(node.args)
        if n_args > 7:
            raise AssertionError(
                f"Unexpected number of input parameters for {self.target}"
            )

        input1_id = self.define_tensor(get_input_node(node, 0), mps_graph)
        output1_id = self.define_tensor(node, mps_graph)

        padding_top, padding_left = [0, 0]
        dilation_height, dilation_width = [1, 1]

        ceil_mode = False
        count_include_pad = True
        divisor_override = 0
        kernel_height, kernel_width = cast(List[int], node.args[1])
        stride_height, stride_width = cast(List[int], node.args[2])
        if n_args >= 4:
            padding_top, padding_left = cast(List[int], node.args[3])
        if n_args >= 5:
            ceil_mode = cast(bool, node.args[4])
        if n_args == 6:
            count_include_pad = cast(bool, node.args[5])
        if n_args == 7:
            divisor_override = cast(int, node.args[6])

        padding_bottom = padding_top * stride_height if ceil_mode else padding_top
        padding_right = padding_left * stride_width if ceil_mode else padding_left

        mps_graph.mps_nodes.append(
            MPSNode(
                mpsnode_union=MPSAvgPool2D(
                    input1_id=input1_id,
                    kernel_height=kernel_height,
                    kernel_width=kernel_width,
                    stride_height=stride_height,
                    stride_width=stride_width,
                    padding_left=padding_left,
                    padding_right=padding_right,
                    padding_top=padding_top,
                    padding_bottom=padding_bottom,
                    dilation_height=dilation_height,
                    dilation_width=dilation_width,
                    ceil_mode=ceil_mode,
                    count_include_pad=count_include_pad,
                    divisor_override=divisor_override,
                    output1_id=output1_id,
                )
            )
        )
