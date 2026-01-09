# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, List

import torch
from executorch.backends.samsung.builders.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.samsung.serialization.enn_graph_schema import EnnGraph
from executorch.backends.transforms import get_shape


@register_node_visitor
class Conv2dVisitor(NodeVisitor):
    target = "aten.convolution.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> None:
        all_input_tensors = []
        input = node.args[0]
        input_id = self.define_tensor(input, enn_graph, vals_to_ids)
        all_input_tensors.append(input_id)

        is_transpose_conv = cast(bool, node.args[6])
        weight_node = node.args[1]
        weight_id = self.define_tensor(
            weight_node,
            enn_graph,
            vals_to_ids,
            is_transpose_conv,
        )
        all_input_tensors.append(weight_id)

        if node.args[2] is not None:
            bias_node = node.args[2]
            bias_id = self.define_tensor(bias_node, enn_graph, vals_to_ids)
            all_input_tensors.append(bias_id)

        stride = cast(List[int], node.args[3])
        padding = cast(List[int], node.args[4])
        dilation = cast(List[int], node.args[5])
        groups = cast(int, node.args[8])
        explicit_padding = [padding[0], padding[1], padding[0], padding[1]]

        input_shape = get_shape(input)
        kernel_shape = get_shape(weight_node)
        params = {}
        self._update_params_qdtype(node, params)
        if "activation" in node.meta:
            params["activation"] = node.meta["activation"]
        params["kernel_h"] = kernel_shape[2]
        params["kernel_w"] = kernel_shape[3]
        params["stride_h"] = stride[0]
        params["stride_w"] = stride[1]
        params["dilation_h"] = dilation[0]
        params["dilation_w"] = dilation[1]
        params["groups"] = groups
        params["padding"] = "EXPLICIT"
        params["padding_type"] = "CONSTANT"  # CONSTANT will be zero-padding
        params["explicit_padding"] = explicit_padding
        params["in_channels"] = input_shape[1]
        params["out_channels"] = kernel_shape[0] * kernel_shape[1] * groups
        params["out_channels"] //= input_shape[1] * input_shape[0]

        output_id = self.define_tensor(node, enn_graph, vals_to_ids)

        is_depthwise_conv = kernel_shape[1] == 1 and kernel_shape[0] / groups == 1
        if is_depthwise_conv:
            conv_type = "DWDECONV2D" if is_transpose_conv else "DWCONV2D"
            enn_graph.define_op(
                node.name, conv_type, all_input_tensors, [output_id], params
            )
        else:
            conv_type = "DECONV2D" if is_transpose_conv else "CONV2D"
            enn_graph.define_op(
                node.name, conv_type, all_input_tensors, [output_id], params
            )
