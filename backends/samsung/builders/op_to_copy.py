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

from executorch.backends.samsung.builders.utils import get_map_dtype, get_tensor
from executorch.backends.samsung.serialization.enn_graph_schema import EnnGraph


@register_node_visitor
class ToCopyVisitor(NodeVisitor):
    target = ["aten._to_copy.default", "dim_order_ops._to_dim_order_copy.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> None:
        memory_format_target = node.kwargs.get("memory_format", torch.contiguous_format)
        to_contiguous = bool(memory_format_target == torch.contiguous_format)
        assert to_contiguous, "Don't support other param in _to_copy"

        input = node.args[0]
        input_id = self.define_tensor(input, enn_graph, vals_to_ids)

        output_id = self.define_tensor(node, enn_graph, vals_to_ids)
        params = {}
        out_tensor = get_tensor(self.exported_program, node)
        params["out_dtype"] = get_map_dtype(out_tensor.dtype)

        enn_graph.define_op(node.name, "CAST", [input_id], [output_id], params)
