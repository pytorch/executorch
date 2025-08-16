# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import Any, List

import torch

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
)
from executorch.backends.arm.tosa_mapping import TosaArg


@register_node_visitor
class CloneDimOrderVisitor(NodeVisitor):
    """
    Implement _clone_dim_order as an identity operation.
    """

    target = "dim_order_ops._clone_dim_order.default"

    tosa_specs = NodeVisitor.tosa_specs

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        import serializer.tosa_serializer as ts  # type: ignore

        validate_num_inputs(self.target, inputs, 1)

        # Since only contiguous dim order is currently supported, treat clone as an identity op.
        tosa_graph.addOperator(ts.TosaOp.Op().IDENTITY, [inputs[0].name], [output.name])
