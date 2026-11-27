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
class AllDimVisitor(NodeVisitor):
    target = "aten.all.dim"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> bool:
        """
        all(input, dim, keepdim=False) -> Tensor

        Maps aten.all.dim directly to Samsung Enn's native REDUCEALL op.

        PyTorch all signature:
            aten::all.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
        - self: input boolean tensor
        - dim: dimension to reduce along
        - keepdim: whether to keep the reduced dimension

        Samsung Enn REDUCEALL op:
        - input: boolean tensor
        - axis: dimension(s) to reduce along
        - keep_dims: whether to keep reduced dimensions
        - output: boolean tensor, True where all elements along dim are True
        """
        input = node.args[0]
        input_id = self.define_tensor(input, enn_graph, vals_to_ids)

        dim = cast(int, node.args[1])
        in_shape = get_shape(input)
        reduce_axes = [dim % len(in_shape)]

        keep_dims = cast(bool, node.args[2]) if len(node.args) > 2 else False
        params = {"keep_dims": keep_dims, "axes": reduce_axes}

        output_id = self.define_tensor(node, enn_graph, vals_to_ids)
        enn_graph.define_op(node.name, "ReduceAll", [input_id], [output_id], params)

        return True
