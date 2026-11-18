# Copyright (c) 2026 Samsung Electronics Co. LTD
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
class OneHotVisitor(NodeVisitor):
    target = "aten.one_hot.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> bool:
        input_indices = node.args[0]
        num_classes = cast(int, node.args[1])

        if num_classes < 0:
            return False

        input_id = self.define_tensor(input_indices, enn_graph, vals_to_ids)

        output_id = self.define_tensor(node, enn_graph, vals_to_ids)

        params = {
            "depth": num_classes,
            "axis": -1,
        }
        self._update_params_qdtype(node, params)

        enn_graph.define_op(node.name, "OneHot", [input_id], [output_id], params)

        return True
