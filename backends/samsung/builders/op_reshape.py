# Copyright (c) 2024 Samsung Electronics Co. LTD
# All rights reserved
from typing import Dict

import torch
from executorch.backends.samsung.builders.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.samsung.builders.utils import get_tensor
from executorch.backends.samsung.serialization.enn_graph_schema import EnnGraph


@register_node_visitor
class ReshapeVisitor(NodeVisitor):
    target = ["aten.view_copy.default", "aten.copy.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> bool:
        is_copy = node.target.__name__ == "aten.copy.default"
        input = node.args[1] if is_copy else node.args[0]

        # node.args[1] may contain "sym_size"
        tensor = get_tensor(self.exported_program, node)
        shape = [1] if len(tensor.size()) == 0 else list(tensor.size())

        if is_copy:
            # copy broadcasts src into dst's shape; RESHAPE only reinterprets
            # the same number of elements, so it can't stand in for a copy
            # whose src isn't already dst's shape.
            src_tensor = get_tensor(self.exported_program, input)
            src_shape = [1] if len(src_tensor.size()) == 0 else list(src_tensor.size())
            if src_shape != shape:
                return False

        input_id = self.define_tensor(input, enn_graph, vals_to_ids)
        output_id = self.define_tensor(node, enn_graph, vals_to_ids)

        enn_graph.define_op(
            node.name, "RESHAPE", [input_id], [output_id], {"new_shape": shape}
        )

        return True
