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
class MulVisitor(NodeVisitor):
    target = "aten.mul.Tensor"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> None:
        input1 = node.args[0]
        input_id_1 = self.define_tensor(input1, enn_graph, vals_to_ids)
        input2 = node.args[1]
        input_id_2 = self.define_tensor(input2, enn_graph, vals_to_ids)

        output_id = self.define_tensor(node, enn_graph, vals_to_ids)

        enn_graph.define_op(node.name, "ELTMUL", [input_id_1, input_id_2], [output_id])
