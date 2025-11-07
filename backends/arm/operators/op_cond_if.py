# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import Any, cast, List

import tosa_serializer as ts

from executorch.backends.arm.operators.node_visitor import (  # type: ignore
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_valid_dtype,
)
from executorch.backends.arm.tosa.mapping import TosaArg  # type: ignore
from executorch.backends.arm.tosa.specification import Tosa_1_00
from torch.fx import Node


@register_node_visitor
class CondVisitor(NodeVisitor):
    target = "cond"

    tosa_specs = NodeVisitor.tosa_specs

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:

        validate_num_inputs(self.target, inputs, 4)
        validate_valid_dtype(self.target, [inputs[0]], ts.DType.BOOL, self.tosa_spec)
        if not isinstance(self.tosa_spec, Tosa_1_00):
            raise ValueError("Trying to lower cond, but TOSA version is <1.0.")
        if not self.tosa_spec.support_extension("cf"):
            raise ValueError(
                f"Trying to lower cond, but TOSA specification {self.tosa_spec} does not support the cf extension."
            )

        attr = ts.TosaSerializerAttribute()
        if_graph, else_graph = (cast(Node, arg).target for arg in node.args[1:3])
        attr.CondIfAttribute(if_graph, else_graph)

        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.COND_IF,
            [
                inputs[0].name,
                *(subgraph_input.name for subgraph_input in inputs[-1].special),
            ],
            [output.name],
            attr,
        )
