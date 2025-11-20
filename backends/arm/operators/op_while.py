# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast, List

import tosa_serializer as ts

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_cf_extension,
    validate_num_inputs,
)
from executorch.backends.arm.tosa.mapping import TosaArg
from torch.fx import Node


@register_node_visitor
class WhileLoopVisitor(NodeVisitor):
    target = "while_loop"

    tosa_specs = NodeVisitor.tosa_specs

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:

        validate_num_inputs(self.target, inputs, 4)
        validate_cf_extension(self.target, self.tosa_spec)

        carried_inputs = inputs[2].special if hasattr(inputs[2], "special") else None
        if carried_inputs is None:
            raise ValueError(f"{self.target}: Expected loop input arguments to be set.")

        additional_inputs = inputs[3].special if hasattr(inputs[3], "special") else None
        if additional_inputs:
            raise ValueError(
                "Additional inputs is not supported, use carried inputs instead."
            )

        attr = ts.TosaSerializerAttribute()
        cond_graph, body_graph = (cast(Node, arg).target for arg in node.args[:2])
        attr.WhileLoopAttribute(cond_graph, body_graph)

        input_names: list[str] = []
        for loop_input in carried_inputs:
            if not isinstance(loop_input, Node):
                raise ValueError(
                    f"{self.target}: Unsupported carried input type {type(loop_input)}."
                )
            input_names.append(loop_input.name)

        if len(input_names) != len(output.multiple_output_names):
            raise ValueError(
                f"TOSA specifies that the number of inputs, {input_names}, need to be the "
                f"same as the number of outputs, {output.multiple_output_names}."
            )

        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.WHILE_LOOP,
            input_names,
            output.multiple_output_names,
            attr,
        )
