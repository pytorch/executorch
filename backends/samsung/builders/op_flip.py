# Copyright (c) 2026 Samsung Electronics Co. LTD
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
class FlipVisitor(NodeVisitor):
    target = "aten.flip.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> bool:
        input = node.args[0]
        input_id = self.define_tensor(input, enn_graph, vals_to_ids)
        axes = node.args[1]
        # Convert list to tensor for axes parameter
        axes_tensor = torch.tensor(axes, dtype=torch.int32)
        axes_id = enn_graph.define_tensor(
            f"{node.name}_axes",
            list(axes_tensor.shape),
            "INT32",
            "CONSTANT",
            data=axes_tensor,
        )

        params = {}
        self._update_params_qdtype(node, params)
        output_id = self.define_tensor(node, enn_graph, vals_to_ids)

        enn_graph.define_op(
            node.name, "ReverseV2", [input_id, axes_id], [output_id], params
        )

        return True
