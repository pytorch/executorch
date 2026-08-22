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
from executorch.backends.samsung.utils.constants import QuantConstants


class _QuantOpVistorBase(NodeVisitor):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> None:
        # input
        input = node.args[0]
        input_id = self.define_tensor(input, enn_graph, vals_to_ids)

        scales = node.args[1]
        if isinstance(scales, torch.Tensor):
            scales = scales.tolist()
        elif not isinstance(scales, list):
            scales = torch.tensor(scales).reshape([1]).tolist()
        zero_points = node.args[2]
        if isinstance(zero_points, torch.Tensor):
            zero_points = zero_points.tolist()
        elif not isinstance(zero_points, list):
            zero_points = torch.tensor(zero_points).reshape([1]).tolist()

        output_id = self.define_tensor(node, enn_graph, vals_to_ids)

        params = {"scales": scales, "zero_points": zero_points}

        if node.target in QuantConstants.QUANT_OPS_KEY_MAP:
            enn_graph.define_op(node.name, "QUANTIZE", [input_id], [output_id], params)
        else:
            enn_graph.define_op(
                node.name, "DEQUANTIZE", [input_id], [output_id], params
            )


@register_node_visitor
class QuantizeVistor(_QuantOpVistorBase):
    target = [
        "quantized_decomposed.quantize_per_tensor.default",
        "quantized_decomposed.quantize_per_channel.default",
    ]
