# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import Any, List

import torch.fx

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_same_dtype,
    validate_valid_dtype,
)

from executorch.backends.arm.tosa.mapping import TosaArg


def unary_operator_factory(unary_target: str, tosa_op):
    "Creates and registers NodeVisitors for operations that have one input and map directly into a TOSA op."

    # Some TOSA unary operators only support float
    fp_only_ops = ["aten.floor.default"]

    class UnaryOperator(NodeVisitor):
        target = unary_target
        tosa_specs = NodeVisitor.tosa_specs

        def __init__(self, *args):
            super().__init__(*args)

        def define_node(
            self,
            node: torch.fx.Node,
            tosa_graph: Any,
            inputs: List[TosaArg],
            output: TosaArg,
        ) -> None:
            import serializer.tosa_serializer as ts  # type: ignore  # noqa: F401

            validate_num_inputs(self.target, inputs, 1)
            validate_same_dtype(self.target, [*inputs, output], ts)

            if self.target in fp_only_ops:
                validate_valid_dtype(
                    self.target,
                    inputs[0],
                    ts.DType.FP32,
                    output.tosa_spec,
                )

            self._serialize_operator(
                node, tosa_graph, tosa_op, [inputs[0].name], [output.name]
            )

    register_node_visitor(UnaryOperator)


import serializer.tosa_serializer as ts  # type: ignore

unary_operator_factory("aten.ceil.default", ts.TosaOp.Op().CEIL)
unary_operator_factory("aten.floor.default", ts.TosaOp.Op().FLOOR)
unary_operator_factory("aten.logical_not.default", ts.TosaOp.Op().LOGICAL_NOT)
