# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import List

import serializer.tosa_serializer as ts
import torch

# pyre-fixme[21]: 'Could not find a module corresponding to import `executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass`.'
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
)
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg

from executorch.backends.arm.tosa_quant_utils import quantize_value
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
    ) -> None:
        attr = ts.TosaSerializerAttribute()

        if inputs[0].dtype == ts.DType.INT8:
            # Get quant parameters
            input_qparams = get_input_qparams(node)  # pyre-ignore[16]
            qargs = input_qparams[0]
            # Convert to quantized representation
            clamp_min_qs = quantize_value(inputs[1].number, qargs)
            clamp_max_qs = quantize_value(inputs[2].number, qargs)
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
