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
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    XNNGraph,
    XNNRope,
    XNode,
)
from executorch.backends.xnnpack.utils.utils import get_input_node

# Register the custom op used by the fusion pass. The fused node targets
# this op after ConvertToRopePass replaces the decomposed HF RoPE subgraph.
lib = torch.library.Library("xnnpack", "FRAGMENT")
lib.define("rope(Tensor input, Tensor weights) -> Tensor")


@torch.library.impl(lib, "rope", "CompositeExplicitAutograd")
def rope_impl(input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    channels = input.shape[-1]
    half_c = channels // 2
    cos = weights[..., :half_c]
    sin = weights[..., half_c:]

    x_real = input[..., :half_c]
    x_imag = input[..., half_c:]

    out_real = x_real * cos - x_imag * sin
    out_imag = x_real * sin + x_imag * cos
    return torch.cat([out_real, out_imag], dim=-1)


@torch.library.impl(lib, "rope", "Meta")
def rope_meta(input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(input)


@register_node_visitor
class RopeVisitor(NodeVisitor):
    target = "rope.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:
        self.define_nodes_tensor_inputs_outputs(node, xnn_graph, vals_to_ids)

        input_id = vals_to_ids[get_input_node(node, 0)]
        weights_id = vals_to_ids[get_input_node(node, 1)]
        output_id = vals_to_ids[node]

        ser_node = XNode(
            xnode_union=XNNRope(
                max_tokens=0,
                input_id=input_id,
                weights_id=weights_id,
                output_id=output_id,
                flags=0,
            ),
            debug_handle=debug_handle,
        )
        xnn_graph.xnodes.append(ser_node)
