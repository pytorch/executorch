# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
from executorch.backends.samsung.builders.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.samsung.serialization.enn_graph_schema import EnnGraph


@register_node_visitor
class MinimumVisitor(NodeVisitor):
    target = "aten.minimum.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> None:
        # input1
        input1 = node.args[0]
        input_id_1 = self.define_tensor(input1, enn_graph, vals_to_ids)

        # input2
        input2 = node.args[1]
        input_id_2 = self.define_tensor(input2, enn_graph, vals_to_ids)

        # output
        output_id = self.define_tensor(node, enn_graph, vals_to_ids)
        vals_to_ids[node] = output_id

        enn_graph.define_op(node.name, "MIN", [input_id_1, input_id_2], [output_id])
