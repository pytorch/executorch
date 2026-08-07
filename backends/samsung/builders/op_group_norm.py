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
class GroupNormVisitor(NodeVisitor):
    target = "aten.native_group_norm.default"

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> None:
        all_input_tensors = []
        input_id = self.define_tensor(node.args[0], enn_graph, vals_to_ids)
        all_input_tensors.append(input_id)

        weight_node = node.args[1]
        weight_id = self.define_tensor(weight_node, enn_graph, vals_to_ids)
        all_input_tensors.append(weight_id)
        bias_node = node.args[2]
        bias_id = self.define_tensor(bias_node, enn_graph, vals_to_ids)
        all_input_tensors.append(bias_id)

        num_groups = cast(int, node.args[6])
        epsilon = node.args[7]

        params = {"num_groups": num_groups, "epsilon": epsilon}

        output_id = self.define_tensor(node, enn_graph, vals_to_ids, output_idx=0)
        enn_graph.define_op(
            node.name, "GROUPNORM", all_input_tensors, [output_id], params
        )
