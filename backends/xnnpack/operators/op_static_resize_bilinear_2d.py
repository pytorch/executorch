# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict

import torch
from executorch.backends.xnnpack.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    XNNGraph,
    XNNStaticResizeBilinear2D,
    XNode,
)
from executorch.backends.xnnpack.utils.utils import get_input_node

from executorch.backends.xnnpack.utils.xnnpack_constants import XNN_FLAG_ALIGN_CORNERS


@register_node_visitor
class StaticResizeBilinear2DVisitor(NodeVisitor):
    target = "aten.upsample_bilinear2d.vec"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:
        self.define_tensor(node, xnn_graph, vals_to_ids, True)
        self.define_tensor(get_input_node(node, 0), xnn_graph, vals_to_ids, True)

        # input
        input_id = vals_to_ids[get_input_node(node, 0)]

        # output
        output_id = vals_to_ids[node]

        new_size = node.meta["val"].shape[-2:]

        flags = XNN_FLAG_ALIGN_CORNERS if cast(bool, node.args[2]) else 0

        ser_node = XNode(
            xnode_union=XNNStaticResizeBilinear2D(
                new_height=new_size[0],
                new_width=new_size[1],
                input_id=input_id,
                output_id=output_id,
                flags=flags,
            ),
            debug_handle=debug_handle,
        )
        xnn_graph.xnodes.append(ser_node)
