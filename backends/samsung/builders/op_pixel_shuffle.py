# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import cast, Dict

import torch
from executorch.backends.samsung.builders.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.samsung.serialization.enn_graph_schema import EnnGraph


@register_node_visitor
class PixelShuffleVisitor(NodeVisitor):
    target = "aten.pixel_shuffle.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> None:
        input_id = self.define_tensor(node.args[0], enn_graph, vals_to_ids)

        scale_factor = cast(int, node.args[1])
        params = {"block_size": scale_factor, "mode": "CRD"}

        output_id = self.define_tensor(node, enn_graph, vals_to_ids)

        enn_graph.define_op(
            node.name, "DEPTH_TO_SPACE", [input_id], [output_id], params
        )
