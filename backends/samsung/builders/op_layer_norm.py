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
class LayerNormVisitor(NodeVisitor):
    target = ["aten.layer_norm.default"]

    def define_node(
            self,
            node: torch.fx.Node,
            enn_graph: EnnGraph,
            vals_to_ids: Dict[torch.Tensor, int],
    ) -> None:
        all_input_tensors = []
        input_node = node.args[0]
        input_id = self.define_tensor(input_node, enn_graph, vals_to_ids)
        all_input_tensors.append(input_id)

        normalized_shapes = node.args[1]
        assert (
            len(normalized_shapes) == 1
            and normalized_shapes[0] == get_shape(input_node)[-1]
        ), "Enn Backend only support norm at last axis."

        weight_node = node.args[2]
        weight_id = self.define_tensor(weight_node, enn_graph, vals_to_ids)
        all_input_tensors.append(weight_id)
        bias_node = node.args[3]
        bias_id = self.define_tensor(bias_node, enn_graph, vals_to_ids)
        all_input_tensors.append(bias_id)

        epsilon = node.args[4] if len(node.args) > 4 else 1e-5
        params = {"epsilon": epsilon}

        output_id = self.define_tensor(node, enn_graph, vals_to_ids)

        enn_graph.define_op(
            node.name, "LAYERNORM", all_input_tensors, [output_id], params
        )
