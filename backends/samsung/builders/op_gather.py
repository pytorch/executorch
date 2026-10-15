# Copyright (c) 2026 Samsung Electronics Co. LTD
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
from executorch.backends.transforms import get_shape


@register_node_visitor
class GatherVisitor(NodeVisitor):
    target = "aten.gather.default"

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> bool:
        input = node.args[0]
        input_id = self.define_tensor(input, enn_graph, vals_to_ids)

        in_shape = get_shape(input)
        axis = cast(int, node.args[1]) % len(in_shape)
        target_indices_node = node.args[2]

        indices_id = self.define_tensor(target_indices_node, enn_graph, vals_to_ids)

        output_id = self.define_tensor(node, enn_graph, vals_to_ids)

        params = {"axis": axis}
        self._update_params_qdtype(node, params)

        enn_graph.define_op(
            node.name, "GATHER", [input_id, indices_id], [output_id], params
        )

        return True
