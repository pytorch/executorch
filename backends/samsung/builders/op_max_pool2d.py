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


@register_node_visitor
class MaxPool2dVisitor(NodeVisitor):
    target = ["aten.max_pool2d.default", "aten.max_pool2d_with_indices.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> None:
        input = node.args[0]
        input_id = self.define_tensor(input, enn_graph, vals_to_ids)

        is_indices = False
        if node.target.__name__ == "aten.max_pool2d_with_indices.default":
            users = list(node.users.keys())

            for user in users:
                if user.target.__name__ == "getitem":
                    getitem_index = user.args[1]
                    is_indices = True
                    if getitem_index != 0:
                        raise AssertionError(
                            "Expected second argument of getitem"
                            f" node for {node.target.__name__ } to be 0, "
                            f"got {getitem_index}. ENN delegate currently "
                            "only supports getting just the max "
                            "values from the op, but doesn't support"
                            " getting the corresponding indices."
                        )

        kernel_size = cast(List[int], node.args[1])
        if len(kernel_size) == 1:
            kernel_size = kernel_size * 2

        stride = cast(List[int], node.args[2]) if len(node.args) > 2 else kernel_size
        if len(stride) == 1:
            stride = stride * 2

        padding = cast(List[int], node.args[3]) if len(node.args) > 3 else [0, 0]
        if len(padding) == 1:
            padding = padding * 2
        explicit_padding = [padding[0], padding[1], padding[0], padding[1]]

        dilation = cast(List[int], node.args[4]) if len(node.args) > 4 else [1, 1]
        if len(dilation) == 1:
            dilation = dilation * 2

        params = {}
        params["kernel_h"] = kernel_size[0]
        params["kernel_w"] = kernel_size[1]
        params["stride_h"] = stride[0]
        params["stride_w"] = stride[1]
        params["padding"] = "EXPLICIT"
        params["explicit_padding"] = explicit_padding
        params["dilation_h"] = dilation[0]
        params["dilation_w"] = dilation[1]

        if len(node.args) > 5:
            ceil_mode = cast(bool, node.args[5])
            assert not ceil_mode, "Not support ceil_mode = True."

        if not is_indices:
            output_id = self.define_tensor(
                node,
                enn_graph,
                vals_to_ids,
            )
        else:
            output_id = self.define_tensor(
                node,
                enn_graph,
                vals_to_ids,
                output_idx=0,
            )

        enn_graph.define_op(node.name, "MAXPOOL2D", [input_id], [output_id], params)
