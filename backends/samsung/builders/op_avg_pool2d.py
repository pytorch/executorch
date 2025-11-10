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
class AvgPool2dVisitor(NodeVisitor):
    target = "aten.avg_pool2d.default"

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

        params = {}
        params["kernel_h"] = kernel_size[0]
        params["kernel_w"] = kernel_size[1]
        params["stride_h"] = stride[0]
        params["stride_w"] = stride[1]
        params["padding"] = "EXPLICIT"
        params["explicit_padding"] = explicit_padding
        self._update_params_qdtype(node, params)

        if len(node.args) > 4:
            ceil_mode = cast(bool, node.args[4])
            assert not ceil_mode, "Not support ceil_mode = True."

        if len(node.args) > 5:
            params["count_include_pad"] = cast(bool, node.args[5])
        else:
            params["count_include_pad"] = True

        if len(node.args) > 6:
            divisor_override = cast(int, node.args[6])
            assert (
                divisor_override == kernel_size[0] * kernel_size[1]
            ), "Not supported divisor_override which is not equal to pooling region."
        output_id = self.define_tensor(node, enn_graph, vals_to_ids)
        enn_graph.define_op(node.name, "AVGPOOL2D", [input_id], [output_id], params)
