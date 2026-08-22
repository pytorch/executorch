#
#  Copyright (c) 2024 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

import logging
from typing import cast

import torch
from executorch.backends.apple.mps.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.apple.mps.serialization.mps_graph_schema import (
    MPSDataType,
    MPSDequantizePerChannelGroup,
    MPSGraph,
    MPSNode,
)
from executorch.backends.apple.mps.utils.mps_utils import get_input_node

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)


@register_node_visitor
class OpDequantizePerChannelGroupDefault(NodeVisitor):
    target = "quantized_decomposed.dequantize_per_channel_group.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        # Weights placeholders shouldn't have been defined until this point
        if get_input_node(node, 0) in self.tensor_to_id:
            raise RuntimeError(
                f"Placeholder for {node.target.__name__} already visited"
            )
        output_id = self.define_tensor(node, mps_graph)
        input_id = self.define_tensor(
            get_input_node(node, 0), mps_graph, MPSDataType.mps_data_type_int4
        )
        scales_id = self.define_tensor(get_input_node(node, 1), mps_graph)

        # there are no zero points in this quantization method (node.args[2] is all zeros)
        zero_points_id = -1
        quant_min = cast(int, node.args[3])
        quant_max = cast(int, node.args[4])
        dtype = self.torch_dtype_to_mps_dtype(node.args[5])
        group_size = cast(int, node.args[6])
        output_dtype = self.torch_dtype_to_mps_dtype(node.args[7])

        dequant_node = MPSNode(
            mpsnode_union=MPSDequantizePerChannelGroup(
                input1_id=input_id,
                output_id=output_id,
                scales_id=scales_id,
                zero_points_id=zero_points_id,
                quant_min=quant_min,
                quant_max=quant_max,
                dtype=dtype,
                group_size=group_size,
                output_dtype=output_dtype,
            )
        )
        mps_graph.mps_nodes.append(dequant_node)


@register_node_visitor
class OpQuantizePerToken(NodeVisitor):
    """
    Dynamic Quantize Per Token Node visitor
    """

    target = "quantized_decomposed.quantize_per_token.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        """
        Skip activation dynamic quantization for now.
        Currently all matmuls are going through [FP16/BF16] @ [QInt4/QInt8].
        Issue: #133407308
        """
        dq_input = self.define_tensor(get_input_node(node, 0), mps_graph)
        self.tensor_to_id[node] = dq_input


@register_node_visitor
class OpDequantizePerToken(NodeVisitor):
    """
    Dequantize Per Token Node visitor
    """

    target = "quantized_decomposed.dequantize_per_token.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        """
        Skip activation dynamic quantization for now.
        Currently all matmuls are going through [FP16/BF16] @ [QInt4/QInt8].
        Issue: #133407308
        """
        dq_input = self.define_tensor(get_input_node(node, 0), mps_graph)
        self.tensor_to_id[node] = dq_input


@register_node_visitor
class OpChooseQparamsToken(NodeVisitor):
    """
    do nothing if node is choose_qparams_per_token_asymmetric.tensor
    """

    target = "quantized_decomposed.choose_qparams_per_token_asymmetric.default"

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        """
        Skip activation dynamic quantization for now.
        Currently all matmuls are going through [FP16/BF16] @ [QInt4/QInt8].
        Issue: #133407308
        """
        input_id = self.define_tensor(get_input_node(node, 0), mps_graph)
        self.tensor_to_id[node] = [input_id, input_id]
