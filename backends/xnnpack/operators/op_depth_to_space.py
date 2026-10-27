# Copyright (c) 2026 Software Mansion
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict

import executorch.backends.xnnpack.custom_ops_lib  # noqa: F401  registers et_xnnpack ops
import torch
from executorch.backends.transforms import get_shape
from executorch.backends.xnnpack.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    XNNDepthToSpace2d,
    XNNGraph,
    XNode,
)
from executorch.backends.xnnpack.utils.utils import check_or_raise, get_input_node


@register_node_visitor
class DepthToSpace2dVisitor(NodeVisitor):
    """
    et_xnnpack.depth_to_space_2d, the block-major depth to space that XNNPACK
    provides natively. Introduced by DecomposeDepthwiseConvTransposePass, see
    custom_ops_lib.py for the channel order.
    """

    target = "et_xnnpack.depth_to_space_2d.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:
        input_node = get_input_node(node, 0)
        self.define_tensor(input_node, xnn_graph, vals_to_ids, convert_to_nhwc=True)
        self.define_tensor(node, xnn_graph, vals_to_ids, convert_to_nhwc=True)

        block_size = cast(int, node.args[1])
        in_shape = get_shape(input_node)
        out_shape = get_shape(node)
        check_or_raise(
            len(in_shape) == 4
            and block_size >= 1
            and in_shape[1] % (block_size * block_size) == 0
            and out_shape[2] == in_shape[2] * block_size
            and out_shape[3] == in_shape[3] * block_size,
            f"depth_to_space_2d(block_size={block_size}) cannot map {in_shape} -> {out_shape}",
        )

        xnn_graph.xnodes.append(
            XNode(
                xnode_union=XNNDepthToSpace2d(
                    block_size=block_size,
                    input_id=vals_to_ids[input_node],
                    output_id=vals_to_ids[node],
                    flags=0,
                ),
                debug_handle=debug_handle,
            )
        )
