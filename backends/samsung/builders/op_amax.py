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
from executorch.backends.transforms import get_shape


@register_node_visitor
class AMaxVisitor(NodeVisitor):
    target = ["aten.amax.default"]

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

        output_id = self.define_tensor(node, enn_graph, vals_to_ids)

        in_shape = get_shape(input)
        dim_arg = node.args[1] if len(node.args) >= 2 else None
        if dim_arg is None:
            reduce_axes = list(range(len(in_shape)))
        elif isinstance(dim_arg, int):
            reduce_axes = [dim_arg % len(in_shape)]
        else:
            reduce_axes = [d % len(in_shape) for d in dim_arg]
        keep_dim = node.args[2] if len(node.args) >= 3 else False
        params = {"keep_dims": keep_dim, "axes": reduce_axes}
        self._update_params_qdtype(node, params)
        enn_graph.define_op(node.name, "ReduceMax", [input_id], [output_id], params)

        return True
