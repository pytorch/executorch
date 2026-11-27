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


@register_node_visitor
class LogicalNotVisitor(NodeVisitor):
    target = "aten.logical_not.default"

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> bool:
        """
        logical_not(input) -> Tensor

        Maps aten.logical_not.default directly to Samsung Enn's native NOT op.

        PyTorch logical_not signature:
            aten::logical_not(Tensor self) -> Tensor
        - self: input tensor (typically bool dtype)
        - output: element-wise logical negation of input

        Samsung Enn NOT op:
        - input: boolean tensor
        - output: element-wise logical NOT of input
        """
        input_id = self.define_tensor(node.args[0], enn_graph, vals_to_ids)

        output_id = self.define_tensor(node, enn_graph, vals_to_ids)

        enn_graph.define_op(node.name, "Not", [input_id], [output_id])

        return True
