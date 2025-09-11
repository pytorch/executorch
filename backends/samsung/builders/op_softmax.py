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
class SoftmaxVisitor(NodeVisitor):
    target = ["aten._softmax.default", "aten._safe_softmax.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ):
        input = node.args[0]
        input_id = self.define_tensor(input, enn_graph, vals_to_ids)

        # output
        output_id = self.define_tensor(node, enn_graph, vals_to_ids)

        axis = cast(int, node.args[1])
        params = {"axis": axis}

        enn_graph.define_op(node.name, "SOFTMAX", [input_id], [output_id], params)
