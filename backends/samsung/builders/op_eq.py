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
class EqVisitor(NodeVisitor):
    target = "aten.eq.Tensor"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> bool:
        """
        eq(input, other) -> Tensor

        Maps aten.eq.Tensor directly to Samsung Enn's native EQUAL op.

        PyTorch eq.Tensor signature:
            aten::eq.Tensor(Tensor self, Tensor other) -> Tensor
        - self: input tensor
        - other: tensor to compare against (broadcastable)
        - output: boolean tensor, True where elements are equal

        Samsung Enn EQUAL op:
        - input1: first tensor
        - input2: second tensor
        - output: boolean tensor, True where elements are equal
        """
        input1 = node.args[0]
        input_id_1 = self.define_tensor(input1, enn_graph, vals_to_ids)

        input2 = node.args[1]
        input_id_2 = self.define_tensor(input2, enn_graph, vals_to_ids)

        output_id = self.define_tensor(node, enn_graph, vals_to_ids)

        enn_graph.define_op(node.name, "Equal", [input_id_1, input_id_2], [output_id])

        return True
