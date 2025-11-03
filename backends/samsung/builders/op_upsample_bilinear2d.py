# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, List

import torch
from executorch.backends.samsung.builders.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.samsung.serialization.enn_graph_schema import EnnGraph
from executorch.backends.transforms import get_shape


@register_node_visitor
class UpsampleBilinear2dVisitor(NodeVisitor):
    target = "aten.upsample_bilinear2d.vec"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> None:
        input = node.args[0]
        input_id = self.define_tensor(input, enn_graph, vals_to_ids)
        in_shape = get_shape(input)
        output_size = cast(List[int], node.args[1])
        scale_factor = [
            output_size[0] * 1.0 / in_shape[-2],
            output_size[1] * 1.0 / in_shape[-1],
        ]

        align_corners = cast(bool, node.args[2])
        if len(node.args) > 3 and node.args[3]:
            scale_factor = cast(List[float], node.args[3])

        params = {
            "align_corners": align_corners,
            "upsampling_factor": scale_factor,
            "half_pixel_centers": True,
        }
        self._update_params_qdtype(node, params)
        output_id = self.define_tensor(node, enn_graph, vals_to_ids)
        enn_graph.define_op(
            node.name, "RESIZE_BILINEAR", [input_id], [output_id], params
        )
