# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import cast, Dict, List

import torch
from executorch.backends.samsung.builders.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.samsung.builders.utils import get_tensor
from executorch.backends.samsung.serialization.enn_graph_schema import EnnGraph
from torch._subclasses import FakeTensor


@register_node_visitor
class ClampVisitor(NodeVisitor):
    target = ["aten.clamp.default", "aten.clamp.Tensor"]

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
        # input_tensor = get_tensor(self.exported_program, input)
        # if input_tensor.dtype == torch.int64:
        #     logging.warning("Currently, int64 clip is unsupported!")
        #     return False

        # The default value of lower bound and upper bound
        clamp_min_default = torch.finfo(torch.float32).min
        clamp_max_default = torch.finfo(torch.float32).max
        output_min = None
        output_max = None

        if len(node.args) > 1 and node.args[1] is not None:
            if isinstance(node.args[1], torch.fx.Node):
                output_min_tensor = get_tensor(self.exported_program, node.args[1])
                if isinstance(output_min_tensor.tolist(), List):
                    logging.warning("Currently, clip min as list is unsupported!")
                    return False
                if not isinstance(output_min_tensor, FakeTensor):
                    output_min = cast(float, output_min_tensor.tolist())
            else:
                output_min = cast(float, node.args[1])
        if len(node.args) > 2 and node.args[2] is not None:
            if isinstance(node.args[2], torch.fx.Node):
                output_max_tensor = get_tensor(self.exported_program, node.args[2])
                if isinstance(output_max_tensor.tolist(), List):
                    logging.warning("Currently, clip max as list is unsupported!")
                    return False
                if not isinstance(output_max_tensor, FakeTensor):
                    output_max = cast(float, output_max_tensor.tolist())
            else:
                output_max = cast(float, node.args[2])

        output_min = (
            output_min if output_min not in (None, -float("inf")) else clamp_min_default
        )
        output_max = (
            output_max if output_max not in (None, float("inf")) else clamp_max_default
        )

        params = {"minimum": output_min, "maximum": output_max}
        self._update_params_qdtype(node, params)

        output_id = self.define_tensor(node, enn_graph, vals_to_ids)

        enn_graph.define_op(node.name, "CLIP", [input_id], [output_id], params)

        return True
