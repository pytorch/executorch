# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import cast, Dict, List

import torch
from executorch.backends.samsung.builders.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.samsung.serialization.enn_graph_schema import EnnGraph
from executorch.backends.transforms import get_shape


@register_node_visitor
class CatVisitor(NodeVisitor):
    target = "aten.cat.default"

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> None:
        tensors = cast(List[torch.fx.Node], node.args[0])
        input_tensor_ids = []

        for in_tensor in tensors:
            input_id = self.define_tensor(in_tensor, enn_graph, vals_to_ids)
            input_tensor_ids.append(input_id)

        in_shape = get_shape(node)
        axis = cast(int, node.args[1]) % len(in_shape) if len(node.args) >= 2 else 0
        params = {"axis": axis}

        output_id = self.define_tensor(node, enn_graph, vals_to_ids)
        enn_graph.define_op(node.name, "CONCAT", input_tensor_ids, [output_id], params)
