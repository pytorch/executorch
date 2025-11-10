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
from executorch.backends.samsung.builders.utils import get_map_dtype
from executorch.backends.samsung.serialization.enn_graph_schema import EnnGraph


@register_node_visitor
class LeakyReluVisitor(NodeVisitor):
    target = ["aten.leaky_relu.default", "aten.prelu.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> None:
        all_input_tensors = []
        input_id = self.define_tensor(node.args[0], enn_graph, vals_to_ids)
        all_input_tensors.append(input_id)

        if node.target.__name__ == "aten.prelu.default":
            negative_slope = node.args[1]
            negative_slope_id = self.define_tensor(
                negative_slope, enn_graph, vals_to_ids
            )
        else:
            negative_slope = cast(float, node.args[1]) if len(node.args) > 1 else 0.01
            negative_slope_tensor = torch.tensor(negative_slope).to(torch.float32)
            negative_slope_node_name = node.name + "_slope"
            dims = list(negative_slope_tensor.size())
            data_type = get_map_dtype(negative_slope_tensor.dtype)
            negative_slope_id = enn_graph.define_tensor(
                negative_slope_node_name,
                dims,
                data_type,
                "CONSTANT",
                negative_slope_tensor.detach().numpy(),
            )

        all_input_tensors.append(negative_slope_id)

        output_id = self.define_tensor(node, enn_graph, vals_to_ids)

        enn_graph.define_op(node.name, "PRELU", all_input_tensors, [output_id])
