# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
from typing import Dict

import torch
from executorch.backends.xnnpack.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    XNNConvert,
    XNNGraph,
    XNode,
)
from executorch.backends.xnnpack.utils.quant_utils import QuantParams
from executorch.backends.xnnpack.utils.utils import get_input_node


@register_node_visitor
class OpDeQuantizePerTensor(NodeVisitor):
    """
    Dequantize Per Tensor Node visitor. We only insert an XNNPACK node if
    this op was found as a graph input or graph output. This is so we
    dequantize the input going in. Every other instance of quantize per
    tensor is only used as signaling for q params of node inputs, so
    we ignore those. This is because xnnpack only supports entire graph
    quantization
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
        We only define a node if it is a graph output
        """
        # TODO:@maxren better handle in-graph quantization conversions, this is hacky
        if self.is_graph_output(node):
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
