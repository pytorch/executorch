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
from executorch.backends.transforms import get_shape


@register_node_visitor
class PixelUnshuffleVisitor(NodeVisitor):
    target = "aten.pixel_unshuffle.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> bool:
        if len(get_shape(node.args[0])) != 4:
            return False

        input_id = self.define_tensor(node.args[0], enn_graph, vals_to_ids)

        downscale_factor = cast(int, node.args[1])
        params = {"block_size": downscale_factor, "mode": "CRD"}

        output_id = self.define_tensor(node, enn_graph, vals_to_ids)

        enn_graph.define_op(
            node.name, "SPACE_TO_DEPTH", [input_id], [output_id], params
        )

        return True
