# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
from executorch.backends.xnnpack._passes.fuse_activation_pass import FuseActivationPass
from executorch.backends.xnnpack.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.xnnpack.operators.quant_params import QuantParams
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    XNNAdd,
    XNNGraph,
    XNode,
)

from executorch.backends.xnnpack.utils.utils import get_input_node


@register_node_visitor
class AddVisitor(NodeVisitor):
    target = "aten.add.Tensor"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:
        # input1
        input1 = get_input_node(node, 0)
        self.define_tensor(
            input1,
            xnn_graph,
            vals_to_ids,
            quant_params=QuantParams.from_inputs(input1, self._exported_program),
        )
        input1_id = vals_to_ids[input1]

        # input2
        input2 = get_input_node(node, 1)
        self.define_tensor(
            input2,
            xnn_graph,
            vals_to_ids,
            quant_params=QuantParams.from_inputs(input2, self._exported_program),
        )
        input2_id = vals_to_ids[input2]

        # output
        output_min_max = FuseActivationPass.get_fused_activation(node)
        self.define_tensor(
            node,
            xnn_graph,
            vals_to_ids,
            quant_params=QuantParams.from_outputs(node),
        )

        output_id = vals_to_ids[node]

        ser_node = XNode(
            xnode_union=XNNAdd(
                input1_id=input1_id, input2_id=input2_id, output_id=output_id, flags=0
            ),
            debug_handle=debug_handle,
            output_min_max=output_min_max,
        )
        xnn_graph.xnodes.append(ser_node)
