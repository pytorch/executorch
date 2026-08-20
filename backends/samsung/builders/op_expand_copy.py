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
class ExpandVisitor(NodeVisitor):
    target = "aten.expand_copy.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ):
        # inputs
        input = node.args[0]
        input_id = self.define_tensor(input, enn_graph, vals_to_ids)

        in_shape = get_shape(input)
        sizes = cast(List[int], node.args[1])
        expand_dims = self.check_expand_dims(sizes, in_shape)

        # output
        output_id = self.define_tensor(node, enn_graph, vals_to_ids)

        if len(expand_dims) == 0:
            params = {"new_shape": [*node.meta["val"].shape]}
            enn_graph.define_op(node.name, "RESHAPE", [input_id], [output_id], params)
        elif len(expand_dims) == 1:
            expand_dim = expand_dims[0]
            params = {"axis": expand_dim}
            enn_graph.define_op(
                node.name,
                "CONCAT",
                [input_id] * sizes[expand_dim],
                [output_id],
                params,
            )
        else:
            raise NotImplementedError("Don't support expanding at more than one axes.")

    def check_expand_dims(self, sizes, in_shape):
        expand_dims = []
        new_size_index = len(sizes)
        in_shape_index = len(in_shape)

        while in_shape_index > 0 and new_size_index > 0:
            in_shape_index -= 1
            new_size_index -= 1
            if (
                sizes[new_size_index] == -1
                or sizes[new_size_index] == in_shape[in_shape_index]
            ):
                continue
            expand_dims.append(in_shape_index)

        while new_size_index > 0:
            new_size_index -= 1
            assert sizes[new_size_index] == 1, "Current expand is unsupported!"

        return expand_dims
