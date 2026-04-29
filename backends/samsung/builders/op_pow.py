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
from executorch.backends.samsung.builders.utils import get_tensor
from executorch.backends.samsung.serialization.enn_graph_schema import EnnGraph


@register_node_visitor
class PowVisitor(NodeVisitor):
    target = "aten.pow.Tensor_Tensor"

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> None:
        input1 = node.args[0]
        input2 = node.args[1]
        input_tensor_1 = get_tensor(self.exported_program, input1)
        input_tensor_2 = get_tensor(self.exported_program, input2)
        assert (
            input_tensor_1.dtype == torch.float32
            and input_tensor_2.dtype == torch.float32
        ), "Requires the two inputs are all float type"

        input_id_1 = self.define_tensor(input1, enn_graph, vals_to_ids)
        input_id_2 = self.define_tensor(input2, enn_graph, vals_to_ids)

        output_id = self.define_tensor(node, enn_graph, vals_to_ids)

        enn_graph.define_op(node.name, "POW", [input_id_1, input_id_2], [output_id])
