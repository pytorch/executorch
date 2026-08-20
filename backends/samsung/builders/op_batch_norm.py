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
class BatchNormVisitor(NodeVisitor):
    target = "aten._native_batch_norm_legit_no_training.default"

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

        weight_node, bias_node, mean_node, var_node = (
            node.args[1],
            node.args[2],
            node.args[3],
            node.args[4],
        )
        weight_id = self.define_tensor(weight_node, enn_graph, vals_to_ids)
        all_input_tensors.append(weight_id)
        bias_id = self.define_tensor(bias_node, enn_graph, vals_to_ids)
        all_input_tensors.append(bias_id)
        mean_id = self.define_tensor(mean_node, enn_graph, vals_to_ids)
        all_input_tensors.append(mean_id)
        var_id = self.define_tensor(var_node, enn_graph, vals_to_ids)
        all_input_tensors.append(var_id)

        eps = node.args[-1]
        params = {"epsilon": eps}

        output_id = self.define_tensor(node, enn_graph, vals_to_ids, output_idx=0)

        enn_graph.define_op(
            node.name, "BatchNormalization", all_input_tensors, [output_id], params
        )
