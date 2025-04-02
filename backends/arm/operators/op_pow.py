# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

import serializer.tosa_serializer as ts
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_specification import TosaSpecification
from serializer.tosa_serializer import TosaOp
from torch.fx import Node


@register_node_visitor
class PowVisitor_080_MI(NodeVisitor):
    target = "aten.pow.Tensor_Tensor"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80+MI"),
    ]

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        if not (inputs[0].dtype == inputs[1].dtype == output.dtype):
            raise ValueError(
                "All inputs and outputs need same dtype."
                f"Got {inputs[0].dtype=}, {inputs[1].dtype=}, {output.dtype=}"
            )
        if inputs[0].dtype not in [ts.DType.FP32, ts.DType.FP16]:
            raise ValueError(
                f"All inputs need to be FP32 or FP16. Got {inputs[0].dtype}"
            )

        tosa_graph.addOperator(
            TosaOp.Op().POW,
            [
                inputs[0].name,
                inputs[1].name,
            ],
            [output.name],
            None,
        )
