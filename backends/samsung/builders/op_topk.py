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
from executorch.backends.transforms import get_shape


@register_node_visitor
class TopKVisitor(NodeVisitor):
    target = "aten.topk.default"

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> None:
        input = node.args[0]
        input_id = self.define_tensor(input, enn_graph, vals_to_ids)

        k = cast(int, node.args[1])
        params = {"k_dims": k}
        in_shape_len = len(get_shape(input))
        dim = cast(int, node.args[2]) if len(node.args) > 2 else in_shape_len - 1
        if dim < 0:
            dim = dim + in_shape_len
        if dim != in_shape_len - 1:
            raise AssertionError("Not supported dim not being last dimension!")

        all_output_tensors = []
        users = list(node.users.keys())
        output_val_idx = 0
        output_val_id = self.define_tensor(
            node,
            enn_graph,
            vals_to_ids,
            output_idx=output_val_idx,
        )
        if len(users) > 0 and users[0].target.__name__ == "getitem":
            vals_to_ids[users[0]] = output_val_id
        all_output_tensors.append(output_val_id)

        output_indices_idx = 1
        output_indices_id = self.define_tensor(
            node,
            enn_graph,
            vals_to_ids,
            output_idx=output_indices_idx,
        )
        if len(users) > 1 and users[1].target.__name__ == "getitem":
            vals_to_ids[users[1]] = output_indices_id
        all_output_tensors.append(output_indices_id)

        if len(node.args) > 3:
            largest = cast(bool, node.args[3])
            if not largest:
                raise AssertionError("Not supported largest = False.")

        if len(node.args) > 4:
            sorted = cast(bool, node.args[4])
            if not sorted:
                raise AssertionError("Not supported sorted = False.")

        enn_graph.define_op(node.name, "TopK", [input_id], all_output_tensors, params)
