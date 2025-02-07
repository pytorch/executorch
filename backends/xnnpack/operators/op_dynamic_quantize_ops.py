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
    is_per_token,
)
from executorch.backends.xnnpack.utils.utils import check_or_raise, get_input_node


@register_node_visitor
class OpDynamicQuantizePerTensor(NodeVisitor):
    """
    Dynamic Quantize Per Tensor Node visitor
    """

    target = "quantized_decomposed.quantize_per_tensor.tensor"

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
        We always define dynamic quantize per tensor nodes because they are always explicit
        """
        q_input = get_input_node(node, 0)

        # fp32 input
        self.define_tensor(q_input, xnn_graph, vals_to_ids)
        input_id = vals_to_ids[q_input]

        # dynamic quantized output
        input_quant_params = QuantParams.from_q_dq_node(node)
        # qinput isn't needed for dynamically quantized nodes since it will always be
        # the output of a convert node. Instead we set q_input to the node itself so
        # we can extract the shape from the dq output
        input_quant_params.q_input = node
        input_quant_params.is_input = False
        check_or_raise(
            input_quant_params.is_dynamic,
            "Internal Error, dynamically quantized node expected dynamic quantized params",
        )
        self.define_tensor(
            node, xnn_graph, vals_to_ids, quant_params=input_quant_params
        )
        output_id = vals_to_ids[node]

        ser_node = XNode(
            xnode_union=XNNConvert(input_id=input_id, output_id=output_id, flags=0),
            debug_handle=debug_handle,
        )
        xnn_graph.xnodes.append(ser_node)


@register_node_visitor
class OpDynamicQuantizePerToken(NodeVisitor):
    """
    Dynamic Quantize Per Token Node visitor
    """

    target = "quantized_decomposed.quantize_per_token.default"

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
        We always define dynamic quantize per token nodes because they are always explicit
        """
        q_input = get_input_node(node, 0)

        # fp32 input
        self.define_tensor(q_input, xnn_graph, vals_to_ids)
        input_id = vals_to_ids[q_input]

        # dynamic quantized output
        input_quant_params = QuantParams.from_q_dq_node(node)
        # qinput isn't needed for dynamically quantized nodes since it will always be
        # the output of a convert node. Instead we set q_input to the node itself so
        # we can extract the shape from the dq output
        input_quant_params.q_input = node
        input_quant_params.is_input = False
        check_or_raise(
            input_quant_params.is_dynamic,
            "Internal Error, dynamically quantized node expected dynamic quantized params",
        )
        self.define_tensor(
            node, xnn_graph, vals_to_ids, quant_params=input_quant_params
        )
        output_id = vals_to_ids[node]

        ser_node = XNode(
            xnode_union=XNNConvert(input_id=input_id, output_id=output_id, flags=0),
            debug_handle=debug_handle,
        )
        xnn_graph.xnodes.append(ser_node)


@register_node_visitor
class OpQuantizeAffine(NodeVisitor):
    target = "quant.quantize_affine.default"

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:
        """
        We always define quantize affine nodes because they are always explicit
        """
        if is_per_channel_group(node):
            # Affine quantized was recognized as per channel group which means that it should
            # be skipped as this means it is used in front of a weight node
            return

        check_or_raise(
            is_per_token(node),
            "Encountered affine quantized op which does not have per-token semantics",
        )
        # Treat this node as dynamic per-token quantization
        q_input = get_input_node(node, 0)

        # fp32 input
        self.define_tensor(q_input, xnn_graph, vals_to_ids)
        input_id = vals_to_ids[q_input]

        # dynamic quantized output
        input_quant_params = QuantParams.from_q_dq_node(node)
        # qinput isn't needed for dynamically quantized nodes since it will always be
        # the output of a convert node. Instead we set q_input to the node itself so
        # we can extract the shape from the dq output
        input_quant_params.q_input = node
        input_quant_params.is_input = False
        check_or_raise(
            input_quant_params.is_dynamic,
            "Internal Error, dynamically quantized node expected dynamic quantized params",
        )
        self.define_tensor(
            node, xnn_graph, vals_to_ids, quant_params=input_quant_params
        )
        output_id = vals_to_ids[node]

        ser_node = XNode(
            xnode_union=XNNConvert(input_id=input_id, output_id=output_id, flags=0),
            debug_handle=debug_handle,
        )
        xnn_graph.xnodes.append(ser_node)
