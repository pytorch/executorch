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
class IndexVisitor(NodeVisitor):
    target = "aten.index.Tensor"

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> None:
        input = node.args[0]
        input_id = self.define_tensor(input, enn_graph, vals_to_ids)

        axis = 0
        valid_indices_node_count = 0
        target_indices_node = None
        for indices_node in node.args[1]:
            if indices_node is not None:
                target_indices_node = indices_node
                valid_indices_node_count += 1
                if valid_indices_node_count > 1:
                    raise NotImplementedError("Not support multi indices node.")
            if target_indices_node is None:
                axis += 1

        indices_id = self.define_tensor(target_indices_node, enn_graph, vals_to_ids)

        output_id = self.define_tensor(node, enn_graph, vals_to_ids)

        params = {"axis": axis}
        enn_graph.define_op(
            node.name, "GATHER", [input_id, indices_id], [output_id], params
        )
