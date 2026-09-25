# Copyright (c) 2025 Samsung Electronics Co. LTD
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
class SplitVisitor(NodeVisitor):
    target = "aten.split_with_sizes_copy.default"

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> bool:
        input = node.args[0]
        input_id = self.define_tensor(input, enn_graph, vals_to_ids)

        # output
        all_output_tensors = []

        copied_indices = []
        for output_idx in range(len(node.args[1])):
            for user in node.users.keys():
                if user.target.__name__ == "getitem" and len(user.args) > 1:
                    copied_idx = user.args[1]
                    copied_indices.append(copied_idx)
                    if copied_idx == output_idx:
                        output_id = self.define_tensor(user, enn_graph, vals_to_ids)
                        all_output_tensors.append(output_id)

        in_shape = get_shape(input)
        points = node.args[1]
        axis = node.args[2] if len(node.args) > 2 else 0
        axis = axis % len(in_shape)

        if len(all_output_tensors) < len(node.args[1]):
            for idx, output_tensor_id in enumerate(all_output_tensors):
                begin = [0] * len(in_shape)
                end = in_shape
                point_idx = copied_indices[idx]
                begin[axis] = sum(points[:point_idx])
                end[axis] = begin[axis] + points[point_idx]
                strides = [1] * len(in_shape)
                params = {
                    "begin": begin,
                    "end": end,
                    "strides": strides,
                    "shrink_axis_mask": pow(2, axis),
                }
                self._update_params_qdtype(node, params)
                enn_graph.define_op(
                    node.name, "STRIDEDSLICE", [input_id], [output_tensor_id], params
                )
        else:
            params = {
                "axis": axis,
                "point": points,
            }
            self._update_params_qdtype(node, params)

            enn_graph.define_op(
                node.name, "SPLIT", [input_id], all_output_tensors, params
            )

        return True
