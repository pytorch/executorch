# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Any, List

import serializer.tosa_serializer as ts

import torch
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_valid_dtype,
)

from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.arm.tosa.mapping import TosaArg


@register_node_visitor
class TableVisitor(NodeVisitor):
    target = "tosa.TABLE.default"

    tosa_specs = [TosaSpecification.create_from_string("TOSA-1.0+INT")]

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        validate_num_inputs(self.target, inputs, 2)
        validate_valid_dtype(
            self.target, inputs, [ts.DType.INT8, ts.DType.INT16], output.tosa_spec
        )
        if inputs[0].dtype == ts.DType.INT8:
            validate_valid_dtype(self.target, output, ts.DType.INT8, output.tosa_spec)
        if inputs[0].dtype == ts.DType.INT16:
            validate_valid_dtype(self.target, output, ts.DType.INT32, output.tosa_spec)

        if inputs[1].name not in self._exported_program.state_dict.keys():  # type: ignore[union-attr]
            raise RuntimeError(
                f"Did not find key {node.name} in state_dict {self._exported_program.state_dict.keys()}."
            )

        table = self._exported_program.state_dict[inputs[1].name]  # type: ignore[union-attr]

        table_tensor_name = node.name + "_table"
        tosa_graph.addConst(
            table.shape,
            ts.DType.INT8 if inputs[0].dtype == ts.DType.INT8 else ts.DType.INT16,
            table.detach().numpy(),
            name=table_tensor_name,
        )

        self._serialize_operator(
            node,
            tosa_graph,
            ts.TosaOp.Op().TABLE,
            [inputs[0].name, table_tensor_name],
            [output.name],
            None,
        )
