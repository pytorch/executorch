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
from executorch.backends.transforms.utils import is_param_node


@register_node_visitor
class PlaceholderVisitor(NodeVisitor):
    """
    To define input tensors.
    This is to make the order of inputs correct.
    """

    target = "placeholder"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> None:
        if is_param_node(self.exported_program, node):
            return
        self.define_tensor(node, enn_graph, vals_to_ids)
