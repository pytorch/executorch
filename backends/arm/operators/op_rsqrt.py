# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List

import torch

import tosa_serializer as ts

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_same_dtype,
    validate_valid_dtype,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.arm.tosa.mapping import TosaArg


@register_node_visitor
class RsqrtVisitor(NodeVisitor):
    target = "aten.rsqrt.default"

    # INT case should be handled by op_table
    tosa_specs = TosaSpecification.all_versions_for_profile("FP")

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        validate_num_inputs(self.target, inputs, 1)
        validate_same_dtype(self.target, [*inputs, output], ts)
        validate_valid_dtype(
            self.target,
            [*inputs, output],
            [ts.DType.FP32, ts.DType.BF16],
            self.tosa_spec,
        )
        attr = ts.TosaSerializerAttribute()
        attr.RsqrtAttribute()
        self._serialize_operator(
            node, tosa_graph, ts.Op.RSQRT, [inputs[0].name], [output.name], attr
        )
