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
    MPSConv2D,
    MPSDepthwiseConv2D,
    MPSGraph,
)
from executorch.backends.apple.mps.utils.mps_utils import get_input_node
from executorch.backends.transforms import get_shape


@register_node_visitor
class Conv2D(NodeVisitor):
    target = "aten.convolution.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        input_shape = get_shape(get_input_node(node, 0))
        weight_shape = get_shape(get_input_node(node, 1))
        groups = cast(int, node.args[8])

        # Convolution is depthwise if groups = input channels and output channel
        # is a positive multiple of input channels

        is_depthwise_conv = (groups > 1 and weight_shape[1] == 1) and (
            len(input_shape) >= 4 and len(weight_shape) >= 4
        )

        mps_node = self.create_tertiary_node(
            node, mps_graph, MPSDepthwiseConv2D if is_depthwise_conv else MPSConv2D
        )

        stride = cast(List[int], node.args[3])
        padding = cast(List[int], node.args[4])
        dilation = cast(List[int], node.args[5])

        if len(stride) == 1:
            stride = [1, stride[0]]
        if len(padding) == 1:
            padding = [0, padding[0]]
        if len(dilation) == 1:
            dilation = [1, dilation[0]]

        mps_node.mpsnode_union.stride_y = stride[0]
        mps_node.mpsnode_union.stride_x = stride[1]
        mps_node.mpsnode_union.dilation_y = dilation[0]
        mps_node.mpsnode_union.dilation_x = dilation[1]
        mps_node.mpsnode_union.groups = groups
        mps_node.mpsnode_union.padding_top = padding[0]
        mps_node.mpsnode_union.padding_bottom = padding[0]
        mps_node.mpsnode_union.padding_right = padding[1]
        mps_node.mpsnode_union.padding_left = padding[1]

        mps_graph.mps_nodes.append(mps_node)
