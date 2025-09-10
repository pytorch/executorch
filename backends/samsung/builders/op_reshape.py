# Copyright (c) 2024 Samsung Electronics Co. LTD
# All rights reserved
from typing import Dict

import torch
from executorch.backends.samsung.builders.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.samsung.serialization.enn_graph_schema import EnnGraph


@register_node_visitor
class ReshapeVisitor(NodeVisitor):
    target = "aten.view_copy.default"

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

        output_id = self.define_tensor(node, enn_graph, vals_to_ids)

        new_shape = node.args[1]
        enn_graph.define_op(
            node.name, "RESHAPE", [input_id], [output_id], {"new_shape": new_shape}
        )
