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
from executorch.backends.transforms.utils import is_param_node
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

        # clamp.default takes Scalar bounds, clamp.Tensor takes constant
        # tensors; a missing or infinite bound means no limit on that side.
        bounds = []
        for idx, default in (
            (1, torch.finfo(torch.float32).min),
            (2, torch.finfo(torch.float32).max),
        ):
            bound = node.args[idx] if len(node.args) > idx else None
            if isinstance(bound, torch.fx.Node):
                bound_tensor = get_tensor(self.exported_program, bound)
                if (
                    not is_param_node(self.exported_program, bound)
                    or bound_tensor.numel() != 1
                ):
                    logging.warning("Currently, clip bounds must be constant scalars!")
                    return False
                # Partitioning runs on a fake copy of the program, whose
                # constants have no values; preprocess sees the real ones.
                bound = (
                    None
                    if isinstance(bound_tensor, FakeTensor)
                    else bound_tensor.item()
                )
            if bound is None or bound in (float("inf"), -float("inf")):
                bound = default
            bounds.append(cast(float, bound))
        output_min, output_max = bounds

        params = {"minimum": output_min, "maximum": output_max}
        self._update_params_qdtype(node, params)

        output_id = self.define_tensor(node, enn_graph, vals_to_ids)

        enn_graph.define_op(node.name, "CLIP", [input_id], [output_id], params)

        return True
