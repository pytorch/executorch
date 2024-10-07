# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
from executorch.backends.xnnpack._passes.fuse_activation_pass import FuseActivationPass
from executorch.backends.xnnpack.operators.node_visitor import (
    get_input_node,
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.xnnpack.operators.quant_params import QuantParams
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    XNNFullyConnected,
    XNNGraph,
    XNode,
)

from executorch.backends.xnnpack.utils.xnnpack_constants import XNN_INVALID_VALUE_ID


@register_node_visitor
class LinearVisitor(NodeVisitor):
    target = "aten.linear.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:

        # input
        input_node = get_input_node(node, 0)
        input_quant_params = QuantParams.from_inputs(input_node, self._exported_program)
        self.define_tensor(
            input_node,
            xnn_graph,
            vals_to_ids,
            quant_params=input_quant_params,
        )
        input_id = vals_to_ids[input_node]

        # filter
        weight_node = get_input_node(node, 1)
        weight_quant_params = QuantParams.from_weights(
            weight_node, self._exported_program
        )
        self.define_tensor(
            weight_node,
            xnn_graph,
            vals_to_ids,
            quant_params=weight_quant_params,
            fp32_static_weights=True,
        )
        filter_id = vals_to_ids[weight_node]

        # bias
        if len(node.args) > 2:
            bias_node = get_input_node(node, 2)
            bias_quant_params = QuantParams.from_bias(
                bias_node, weight_quant_params, input_quant_params
            )
            self.define_tensor(
                get_input_node(node, 2),
                xnn_graph,
                vals_to_ids,
                quant_params=bias_quant_params,
                fp32_static_weights=True,
            )
            bias_id = vals_to_ids[bias_node]
        else:
            bias_id = XNN_INVALID_VALUE_ID

        # output
        output_min_max = FuseActivationPass.get_fused_activation(node)
        output_quant_params = QuantParams.from_outputs(node)
        self.define_tensor(
            node,
            xnn_graph,
            vals_to_ids,
            quant_params=output_quant_params,
        )
        output_id = vals_to_ids[node]

        ser_node = XNode(
            xnode_union=XNNFullyConnected(
                input1_id=input_id,
                filter_id=filter_id,
                bias_id=bias_id,
                output_id=output_id,
                flags=0,
            ),
            debug_handle=debug_handle,
            output_min_max=output_min_max,
        )
        xnn_graph.xnodes.append(ser_node)
