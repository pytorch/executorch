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
from executorch.backends.transforms import get_shape


@register_node_visitor
class LinearVisitor(NodeVisitor):
    target = "aten.linear.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> None:
        all_input_tensors = []
        input = node.args[0]
        input_id = self.define_tensor(input, enn_graph, vals_to_ids)
        all_input_tensors.append(input_id)

        weight_node = node.args[1]
        weight_id = self.define_tensor(weight_node, enn_graph, vals_to_ids)
        all_input_tensors.append(weight_id)

        if len(node.args) > 2 and node.args[2] is not None:
            bias_node = node.args[2]
            bias_id = self.define_tensor(bias_node, enn_graph, vals_to_ids)
            all_input_tensors.append(bias_id)

        weight_shape = get_shape(weight_node)
        params = {"in_channels": weight_shape[1], "out_channels": weight_shape[0]}

        output_id = self.define_tensor(node, enn_graph, vals_to_ids)

        enn_graph.define_op(node.name, "FC", all_input_tensors, [output_id], params)
