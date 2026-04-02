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
class SplitVisitor(NodeVisitor):
    target = "aten.split_with_sizes_copy.default"

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ):
        input = node.args[0]
        input_id = self.define_tensor(input, enn_graph, vals_to_ids)

        # output
        all_output_tensors = []

        for output_idx in range(len(node.args[1])):
            output_id = self.define_tensor(
                node,
                enn_graph,
                vals_to_ids,
                output_idx=output_idx,
            )
            all_output_tensors.append(output_id)

        axis = node.args[2] if len(node.args) > 2 else 0

        params = {}
        params["axis"] = axis
        params["point"] = node.args[1]

        enn_graph.define_op(node.name, "SPLIT", [input_id], all_output_tensors, params)
