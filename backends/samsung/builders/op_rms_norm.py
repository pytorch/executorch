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
from executorch.backends.samsung.builders.utils import get_tensor
from executorch.backends.samsung.serialization.enn_graph_schema import EnnGraph


@register_node_visitor
class RmsNormVisitor(NodeVisitor):
    target = "aten.rms_norm.default"

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> None:
        # args of node : ['input', 'normalized_shape', 'weight', 'eps']
        input = node.args[0]
        input_id = self.define_tensor(input, enn_graph, vals_to_ids)

        # input2
        normalized_shape = cast(List[int], node.args[1])

        gamma_node = node.args[2]
        gamma_id = self.define_tensor(gamma_node, enn_graph, vals_to_ids)

        epsilon = node.args[3]
        if isinstance(epsilon, torch.fx.Node):
            epsilon = get_tensor(self.exported_program, epsilon)
            epsilon = epsilon.item()

        params = {}
        params["normalize_shape"] = normalized_shape
        params["param_num"] = 2
        params["epsilon"] = epsilon

        output_id = self.define_tensor(node, enn_graph, vals_to_ids)

        enn_graph.define_op(
            node.name, "RMSNORM", [input_id, gamma_id], [output_id], params
        )
