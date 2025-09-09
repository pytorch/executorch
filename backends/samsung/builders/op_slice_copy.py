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
class SliceCopyVisitor(NodeVisitor):
    target = "aten.slice_copy.Tensor"

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

        in_shape = get_shape(input)
        dim = cast(int, node.args[1])
        if dim < 0:
            dim = dim + len(in_shape)
        start_val = cast(int, node.args[2])
        if start_val < 0:
            start_val = start_val + in_shape[dim]
        end_val = min(cast(int, node.args[3]), in_shape[dim])
        if end_val < 0:
            end_val = end_val + in_shape[dim]

        step = cast(int, node.args[4]) if len(node.args) > 4 else 1

        begin = [0] * len(in_shape)
        begin[dim] = start_val
        end = in_shape
        end[dim] = end_val
        strides = [1] * len(in_shape)
        strides[dim] = step

        params = {"begin": begin, "end": end, "strides": strides}

        enn_graph.define_op(node.name, "STRIDEDSLICE", [input_id], [output_id], params)
