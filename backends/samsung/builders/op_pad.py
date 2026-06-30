# Copyright (c) 2026 Samsung Electronics Co. LTD
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
from executorch.backends.samsung.utils.constants import QuantConstants
from executorch.backends.transforms import get_shape


@register_node_visitor
class PadVisitor(NodeVisitor):
    target = "aten.pad.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> bool:

        input1 = node.args[0]
        input_id_1 = self.define_tensor(input1, enn_graph, vals_to_ids)

        padding = cast(List[int], node.args[1])

        # padding from last dim, here stores padding from first dim and fill 0 at the beginning.
        # Reverse padding in pairs of two elements
        padding = [
            x for i in range(len(padding) - 2, -2, -2) for x in padding[i : i + 2]
        ]
        input_shape = get_shape(input1)
        padding = [0] * (len(input_shape) * 2 - len(padding)) + padding
        # Rearrange padding: even indices first, then odd indices
        padding = [padding[i] for i in range(0, len(padding), 2)] + [
            padding[i] for i in range(1, len(padding), 2)
        ]

        mode = "constant"
        constant_value = 0
        if len(node.args) > 2:
            mode = node.args[2]
        if mode == "constant":
            quant_attrs = node.meta.get("quantize_attrs")
            if quant_attrs is not None:
                zero_points = EnnGraph._affine_meta_param(
                    quant_attrs[QuantConstants.QUANT_KEY.zero_point]
                )
                if len(zero_points) == 1:
                    constant_value = zero_points[0]

        params = {"pads": padding, "mode": mode, "constant_value": constant_value}
        self._update_params_qdtype(node, params)
        output_id = self.define_tensor(node, enn_graph, vals_to_ids)

        enn_graph.define_op(node.name, "Pad", [input_id_1], [output_id], params)

        return True
