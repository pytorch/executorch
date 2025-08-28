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
class ClampVisitor(NodeVisitor):
    target = "aten.clamp.default"

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

        # The default value of lower bound and upper bound
        output_min = torch.finfo(torch.float32).min
        output_max = torch.finfo(torch.float32).max
        if node.args[1] is not None:
            output_min = cast(float, node.args[1])
        if len(node.args) > 2 and node.args[2] is not None:
            output_max = cast(float, node.args[2])

        params = {"minimum": output_min, "maximum": output_max}
        output_id = self.define_tensor(node, enn_graph, vals_to_ids)

        enn_graph.define_op(node.name, "CLIP", [input_id], [output_id], params)
