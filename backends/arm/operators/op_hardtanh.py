# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import List

import serializer.tosa_serializer as ts
import torch
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg

from executorch.backends.arm.tosa_quant_utils import get_quant_node_args
from serializer.tosa_serializer import TosaOp


@register_node_visitor
class HardTanhVisitor(NodeVisitor):
    target = "aten.hardtanh.default"

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
        is_quant_node: bool,
    ) -> None:
        attr = ts.TosaSerializerAttribute()

        if is_quant_node:
            # Get quant parameters
            scale, zp, qmin, qmax = get_quant_node_args(node.all_input_nodes[0])
            # Convert to quantized representation
            clamp_min_qs = round((inputs[1].number / scale) + zp)
            clamp_min_qs = max(clamp_min_qs, qmin)
            clamp_max_qs = round((inputs[2].number / scale) + zp)
            clamp_max_qs = min(clamp_max_qs, qmax)
            # Set fp values to 0.0 since they are not used
            clamp_min_fp = 0.0
            clamp_max_fp = 0.0
        else:
            clamp_min_fp = inputs[1].number
            clamp_max_fp = inputs[2].number
            # Set qs values to 0 since they are not used
            clamp_min_qs = 0
            clamp_max_qs = 0

        attr.ClampAttribute(
            tosa_graph.builder,
            clamp_min_qs,
            clamp_max_qs,
            clamp_min_fp,
            clamp_max_fp,
        )

        tosa_graph.addOperator(TosaOp.Op().CLAMP, [inputs[0].name], [output.name], attr)
