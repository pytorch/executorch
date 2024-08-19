# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict

import torch
from executorch.backends.transforms import get_shape
from executorch.backends.xnnpack.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    XNNGraph,
    XNNScaledDotProductAttention,
    XNode,
)
from executorch.backends.xnnpack.utils.utils import get_input_node


@register_node_visitor
class SDPAVisitor(NodeVisitor):
    target = "aten.scaled_dot_product_attention.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    @staticmethod
    def get_fake_attr(name: str, value: torch.Tensor) -> torch.fx.Node:
        g = torch.fx.Graph()
        gm = torch.fx.GraphModule({}, g)
        fake_node = torch.fx.Node(g, name, "get_attr", target=name, args=(), kwargs={})
        g._owning_module = gm
        setattr(g._owning_module, name, value)
        fake_node.meta["val"] = value
        return fake_node

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:
        # inputs
        for i in range(0, 4):
            inp = get_input_node(node, i)
            self.define_tensor(
                inp,
                xnn_graph,
                vals_to_ids,
            )

        # Make sure mask is not bool
        mask_node = get_input_node(node, 3)
        mask_dtype = mask_node.meta["val"].dtype
        assert mask_dtype in [
            torch.float,
            torch.float16,
        ], "SDPA Mask must be a float (or half) tensor"

        # Make sure mask is not >2D
        assert len(get_shape(mask_node)) == 2, "SDPA Mask must be 2D"

        # Hack to broadcast the scale
        q_shape = get_shape(get_input_node(node, 0))
        embedding_dim = q_shape[-1]
        scale = 1 / (embedding_dim**0.5)
        if "scale" in node.kwargs and node.kwargs["scale"]:
            scale = cast(float, node.kwargs["scale"])

        t = torch.full((embedding_dim,), scale, dtype=mask_dtype)
        scale_node = self.get_fake_attr("scale", t)
        self.define_tensor(
            scale_node,
            xnn_graph,
            vals_to_ids,
        )

        # outputs
        outp = node
        self.define_tensor(
            outp,
            xnn_graph,
            vals_to_ids,
        )

        # ids
        q_id = vals_to_ids[get_input_node(node, 0)]
        k_id = vals_to_ids[get_input_node(node, 1)]
        v_id = vals_to_ids[get_input_node(node, 2)]
        mask_id = vals_to_ids[mask_node]
        scale_id = vals_to_ids[scale_node]
        output_id = vals_to_ids[outp]

        # Create a new node
        sdpa_node = XNode(
            xnode_union=XNNScaledDotProductAttention(
                query_id=q_id,
                key_id=k_id,
                value_id=v_id,
                scale_id=scale_id,
                mask_id=mask_id,
                output_id=output_id,
                flags=0,
            ),
            debug_handle=debug_handle,
        )
        xnn_graph.xnodes.append(sdpa_node)
