# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
from executorch.backends.xnnpack.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.xnnpack.operators.quant_params import QuantParams
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    XNNConvert,
    XNNGraph,
    XNode,
)
from executorch.backends.xnnpack.utils.quant_utils import (
    is_per_channel_group,
    is_tagged_as_implicit_q_dq,
    validate_quant_scales,
    validate_quant_zeropoints,
)
from executorch.backends.xnnpack.utils.utils import get_input_node, get_param_tensor


class OpStaticQDQNode(NodeVisitor):
    def check_scales_zeropoints(self, node) -> None:
        scales = node.args[1]
        zero_points = node.args[2]
        is_groupwise = is_per_channel_group(node)
        dtype = node.args[-1]
        if is_groupwise:
            dtype = node.args[-3]

        if isinstance(scales, torch.fx.Node):
            scales = get_param_tensor(self.exported_program, scales)

        if isinstance(zero_points, torch.fx.Node):
            zero_points = get_param_tensor(self.exported_program, zero_points)

        try:
            validate_quant_scales(scales)
            validate_quant_zeropoints(zero_points, dtype, is_groupwise)
        except ValueError as e:
            raise ValueError(
                f"Invalid quantization scale or zero point for {node}: {e}"
            )

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:
        # check scales and zp are valid
        self.check_scales_zeropoints(node)


@register_node_visitor
class OpDeQuantizePerTensor(OpStaticQDQNode):
    """
    Dequantize Per Tensor Node visitor
    """

    target = "quantized_decomposed.dequantize_per_tensor.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:
        """
        We only define a node if it is not an implict dq node
        """
        # check scales and zp are valid
        super().define_node(node, xnn_graph, vals_to_ids, debug_handle)

        if not is_tagged_as_implicit_q_dq(node):
            dq_input = get_input_node(node, 0)
            input_quant_params = QuantParams.from_q_dq_node(node)
            # fp32 output
            self.define_tensor(node, xnn_graph, vals_to_ids)
            output_id = vals_to_ids[node]

            # qint8 input
            input_quant_params.is_output = False
            self.define_tensor(
                dq_input, xnn_graph, vals_to_ids, quant_params=input_quant_params
            )
            input_id = vals_to_ids[dq_input]

            ser_node = XNode(
                xnode_union=XNNConvert(input_id=input_id, output_id=output_id, flags=0),
                debug_handle=debug_handle,
            )
            xnn_graph.xnodes.append(ser_node)
        else:
            # If this node was ignored, then its id is the same as its parent
            dq_input = get_input_node(node, 0)
            if dq_input in vals_to_ids:
                vals_to_ids[node] = vals_to_ids[dq_input]


@register_node_visitor
class OpQuantizePerTensor(OpStaticQDQNode):
    """
    Quantize Per Tensor Node visitor
    """

    target = "quantized_decomposed.quantize_per_tensor.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:
        """
        We only define a node if it is not an implict q node
        """
        # check scales and zp are valid
        super().define_node(node, xnn_graph, vals_to_ids, debug_handle)

        q_input = get_input_node(node, 0)
        if not is_tagged_as_implicit_q_dq(node):
            input_quant_params = QuantParams.from_q_dq_node(node)
            # fp32 input
            self.define_tensor(q_input, xnn_graph, vals_to_ids)
            input_id = vals_to_ids[q_input]

            # qint8 output
            input_quant_params.q_input = node
            input_quant_params.is_input = False
            self.define_tensor(
                node, xnn_graph, vals_to_ids, quant_params=input_quant_params
            )
            output_id = vals_to_ids[node]

            ser_node = XNode(
                xnode_union=XNNConvert(input_id=input_id, output_id=output_id, flags=0),
                debug_handle=debug_handle,
            )
            xnn_graph.xnodes.append(ser_node)
        else:
            # If this node was ignored, then its id is the same as its parents
            if q_input in vals_to_ids:
                vals_to_ids[node] = vals_to_ids[q_input]


@register_node_visitor
class OpDequantizePerChannelDefault(OpStaticQDQNode):
    """
    do nothing if node is dequantize_per_channel.default
    """

    target = "quantized_decomposed.dequantize_per_channel.default"


@register_node_visitor
class OpQuantizePerChannelDefault(OpStaticQDQNode):
    """
    do nothing if node is quantize_per_channel.default
    """

    target = "quantized_decomposed.quantize_per_channel.default"


@register_node_visitor
class OpQuantizePerChannelGroupDefault(OpStaticQDQNode):
    """
    do nothing if node is quantize_per_channel_group.default
    """

    target = "quantized_decomposed.quantize_per_channel_group.default"


@register_node_visitor
class OpDequantizePerChannelGroupDefault(OpStaticQDQNode):
    """
    do nothing if node is dequantize_per_channel_group.default
    """

    target = "quantized_decomposed.dequantize_per_channel_group.default"
