# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast, List

import tosa_serializer as ts
from executorch.backends.arm._passes.arm_pass_utils import get_output_dim_orders

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_cf_extension,
    validate_num_inputs,
)
from executorch.backends.arm.tosa.mapping import map_dtype, TosaArg
from executorch.backends.arm.tosa.utils import tosa_shape

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
        cond_graph, body_graph = (str(cast(Node, arg).target) for arg in node.args[:2])
        attr.WhileLoopAttribute(cond_graph, body_graph)

        input_names: list[str] = []
        for loop_input in carried_inputs:
            if not isinstance(loop_input, Node):
                raise ValueError(
                    f"{self.target}: Unsupported carried input type {type(loop_input)}."
                )
            input_names.append(loop_input.name)

        num_inputs = len(input_names)
        num_outputs = len(output.multiple_output_names)
        if num_inputs > num_outputs:
            # If we have more inputs than outputs, we can just add missing output tensors.
            body_module = getattr(node.graph.owning_module, body_graph)
            output_dim_orders = get_output_dim_orders(body_module)
            body_outputs = body_module.graph.output_node().args[0]
            outputs_needing_tensors = body_outputs[num_outputs - num_inputs :]
            output_dim_orders = output_dim_orders[num_outputs - num_inputs :]
            for (
                output_needing_tensor,
                dim_order,
            ) in zip(outputs_needing_tensors, output_dim_orders, strict=True):
                tensor_name = output_needing_tensor.name + "_dummy"
                shape = output_needing_tensor.meta["val"].shape
                dtype = map_dtype(output_needing_tensor.meta["val"].dtype)

                tosa_graph.currRegion.currBasicBlock.addTensor(
                    tensor_name,
                    tosa_shape(shape, dim_order),
                    dtype,
                )
                output.multiple_output_names.append(tensor_name)
        elif num_inputs < num_outputs:
            # This is a strange case, if we reach it something bad has happened.
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
