# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import cast, Dict

import torch
from executorch.backends.samsung.builders.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.samsung.builders.utils import get_tensor
from executorch.backends.samsung.serialization.enn_graph_schema import EnnGraph


@register_node_visitor
class ClampVisitor(NodeVisitor):
    target = "aten.clamp.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> bool:
        input = node.args[0]
        input_id = self.define_tensor(input, enn_graph, vals_to_ids)
        input_tensor = get_tensor(self.exported_program, input)
        if input_tensor.dtype == torch.int64:
            logging.warning("Currently, int64 clip is unsupported!")
            return False

        # The default value of lower bound and upper bound
        output_min = torch.finfo(torch.float32).min
        output_max = torch.finfo(torch.float32).max

        if node.args[1] is not None:
            output_min = cast(float, node.args[1])
        if len(node.args) > 2 and node.args[2] is not None:
            output_max = cast(float, node.args[2])

        params = {"minimum": output_min, "maximum": output_max}
        self._update_params_qdtype(node, params)

        output_id = self.define_tensor(node, enn_graph, vals_to_ids)

        enn_graph.define_op(node.name, "CLIP", [input_id], [output_id], params)

        return True
