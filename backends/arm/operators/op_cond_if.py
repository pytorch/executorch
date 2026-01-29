# Copyright 2025-2026 Arm Limited and/or its affiliates.
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
    validate_cf_extension,
    validate_num_inputs,
    validate_valid_dtype,
)
from executorch.backends.arm.tosa.mapping import TosaArg  # type: ignore
from torch.fx import Node


@register_node_visitor
class CondVisitor(NodeVisitor):
    target = "cond"

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:

        validate_num_inputs(self.target, inputs, 4)
        validate_valid_dtype(self.target, [inputs[0]], ts.DType.BOOL, self.tosa_spec)
        validate_cf_extension(self.target, self.tosa_spec)

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
            output.multiple_output_names,
            attr,
        )
