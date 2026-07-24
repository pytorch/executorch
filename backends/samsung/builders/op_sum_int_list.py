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
class SumDimIntListVisitor(NodeVisitor):
    target = "aten.sum.dim_IntList"

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> None:
        input = node.args[0]
        input_id = self.define_tensor(input, enn_graph, vals_to_ids)

        reduce_axes = cast(List[int], node.args[1])
        in_shape = get_shape(input)
        reduce_axes = [axis % len(in_shape) for axis in reduce_axes]

        keep_dims = cast(bool, node.args[2]) if len(node.args) > 2 else False
        params = {"keep_dims": keep_dims, "axis": reduce_axes}

        output_id = self.define_tensor(node, enn_graph, vals_to_ids)
        enn_graph.define_op(node.name, "REDUCESUM", [input_id], [output_id], params)
