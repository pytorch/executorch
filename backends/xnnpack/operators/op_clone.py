# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Dict

import torch
from executorch.backends.xnnpack.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.xnnpack.operators.quant_params import QuantParams
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    XNNCopy,
    XNNGraph,
    XNode,
)
from executorch.backends.xnnpack.utils.utils import get_input_node


@register_node_visitor
class CloneVisitor(NodeVisitor):
    target = "aten.clone.default"

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:
        self.define_tensor(
            node,
            xnn_graph,
            vals_to_ids,
            quant_params=QuantParams.from_outputs(node),
        )
        input_node = get_input_node(node, 0)
        self.define_tensor(
            input_node,
            xnn_graph,
            vals_to_ids,
            quant_params=QuantParams.from_inputs(input_node, self._exported_program),
        )

        # Sanity check that the input and output dim order are the same. We don't
        # handle dim order conversions yet.
        dim_order = node.kwargs.get("dim_order", None)
        input_meta = node.args[0].meta["val"]
        assert dim_order is None or list(input_meta.dim_order()) == dim_order

        # input
        input_id = vals_to_ids[get_input_node(node, 0)]

        # output
        output_id = vals_to_ids[node]

        ser_node = XNode(
            xnode_union=XNNCopy(
                input_id=input_id,
                output_id=output_id,
                flags=0,
            ),
            debug_handle=debug_handle,
        )
        xnn_graph.xnodes.append(ser_node)
