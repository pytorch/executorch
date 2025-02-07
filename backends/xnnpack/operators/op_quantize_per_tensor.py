# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
from executorch.backends.xnnpack._passes.tag_implicit_q_dq_pass import (
    TagImplicitQDqPass,
)
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
from executorch.backends.xnnpack.utils.utils import get_input_node


@register_node_visitor
class OpQuantizePerTensor(NodeVisitor):
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
        q_input = get_input_node(node, 0)
        if not TagImplicitQDqPass.is_tagged_as_implicit_q_dq(node):
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
