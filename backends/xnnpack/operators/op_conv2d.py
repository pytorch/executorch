# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, List

import torch
from executorch.backends.transforms import get_shape
from executorch.backends.xnnpack._passes.fuse_activation_pass import FuseActivationPass
from executorch.backends.xnnpack.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.xnnpack.operators.quant_params import QuantParams
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    XNNConv2d,
    XNNDepthwiseConv2d,
    XNNGraph,
    XNode,
)
from executorch.backends.xnnpack.utils.utils import check_or_raise, get_input_node

from executorch.backends.xnnpack.utils.xnnpack_constants import XNN_INVALID_VALUE_ID


@register_node_visitor
class Conv2d(NodeVisitor):
    target = "aten.convolution.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:
        kwargs = {}
        # input
        input_node = get_input_node(node, 0)
        input_quant_params = QuantParams.from_inputs(input_node, self._exported_program)
        self.define_tensor(
            input_node,
            xnn_graph,
            vals_to_ids,
            convert_to_nhwc=True,
            quant_params=input_quant_params,
        )  # NHWC input
        kwargs["input1_id"] = vals_to_ids[get_input_node(node, 0)]

        # filter shape for pytorch convolution is (oc, inc/groups, height, width)
        # shape for xnnpack convolution is (oc, height, width, inc/groups), to convert
        # to the proper shape, this is essentially a NCHW to NHWC conversion
        kernel_node = get_input_node(node, 1)
        kernel_shape = get_shape(kernel_node)
        groups = cast(int, node.args[8])
        group_input_channels = kernel_shape[1]
        group_output_channels = int(kernel_shape[0] / groups)

        # XNNPACK expects the kernel's N and C dimensions to be swapped for
        # Depthwise Convolution, which occurs under the following conditions:
        # 1) groups = input_channels (i.e. group_input_channels = 1)
        # 2) output_channels is a positive integer multiple of input channels
        is_depthwise_conv = (group_input_channels == 1) and (
            group_output_channels % group_input_channels == 0
        )
        weight_quant_params = QuantParams.from_weights(
            kernel_node, self._exported_program
        )
        fp32_static_weights = kernel_node.meta["val"].dtype == torch.float16

        self.define_tensor(
            kernel_node,
            xnn_graph,
            vals_to_ids,
            convert_to_nhwc=True,
            swap_nc_for_depthwise_weights=is_depthwise_conv,
            quant_params=weight_quant_params,
            fp32_static_weights=fp32_static_weights,
        )
        kwargs["filter_id"] = vals_to_ids[get_input_node(node, 1)]

        # output
        output_min_max = FuseActivationPass.get_fused_activation(node)
        output_quant_params = QuantParams.from_outputs(node)
        self.define_tensor(
            node,
            xnn_graph,
            vals_to_ids,
            convert_to_nhwc=True,
            quant_params=output_quant_params,
        )  # NHWC output
        kwargs["output_id"] = vals_to_ids[node]

        # bias
        kwargs["bias_id"] = XNN_INVALID_VALUE_ID
        if node.args[2] is not None:
            # If there is a bias
            bias_node = get_input_node(node, 2)
            bias_quant_params = QuantParams.from_bias(
                bias_node, weight_quant_params, input_quant_params
            )
            self.define_tensor(
                get_input_node(node, 2),
                xnn_graph,
                vals_to_ids,
                convert_to_nhwc=False,
                quant_params=bias_quant_params,
                fp32_static_weights=fp32_static_weights,
            )
            kwargs["bias_id"] = vals_to_ids[get_input_node(node, 2)]

        stride = cast(List[int], node.args[3])
        padding = cast(List[int], node.args[4])
        dilation = cast(List[int], node.args[5])
        if len(padding) == 1:
            padding = padding + padding

        # args[6] = transposed
        check_or_raise(
            not cast(bool, node.args[6]), "No support for transposed convolution"
        )
        # args[7] = output padding
        check_or_raise(
            all(out_pad == 0 for out_pad in cast(List[int], node.args[7])),
            "XNNPACK does not support output padding",
        )

        check_or_raise(
            len(stride) == 2, "XNNPACK currently only supports 2D convolution"
        )
        kwargs["padding_top"] = padding[0]
        kwargs["padding_right"] = padding[1]
        kwargs["padding_bottom"] = padding[0]
        kwargs["padding_left"] = padding[1]
        kwargs["kernel_height"] = kernel_shape[2]
        kwargs["kernel_width"] = kernel_shape[3]
        kwargs["subsampling_height"] = stride[0]
        kwargs["subsampling_width"] = stride[1]
        kwargs["dilation_height"] = dilation[0]
        kwargs["dilation_width"] = dilation[1]
        kwargs["group_input_channels"] = group_input_channels
        kwargs["group_output_channels"] = group_output_channels
        kwargs["groups"] = groups
        kwargs["adjustment_height"] = 0
        kwargs["adjustment_width"] = 0
        kwargs["flags"] = 0

        if is_depthwise_conv:
            conv_node_type = XNNDepthwiseConv2d
        else:
            conv_node_type = XNNConv2d

        ser_node = XNode(
            xnode_union=conv_node_type(
                **kwargs,
            ),
            debug_handle=debug_handle,
            output_min_max=output_min_max,
        )
        xnn_graph.xnodes.append(ser_node)
