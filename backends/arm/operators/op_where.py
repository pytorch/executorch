# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Sequence

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)

from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_same_dtype,
    validate_valid_dtype,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_specification import TosaSpecification
from torch.fx import Node


@register_node_visitor
class WhereVisitor_INT(NodeVisitor):
    target = "aten.where.self"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
    ]

    def __init__(self, *args):
        super().__init__(*args)

    def _add_node_to_tosa_graph(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
        supported_dtypes: Sequence,
    ) -> None:
        import serializer.tosa_serializer as ts

        validate_num_inputs(self.target, inputs, 3)
        # Not first input, which is condition tensor.
        validate_same_dtype(self.target, inputs[1:], ts)
        validate_valid_dtype(self.target, inputs[0], ts.DType.BOOL, output.tosa_spec)
        validate_valid_dtype(
            self.target,
            [*inputs[1:], output],
            supported_dtypes,
            output.tosa_spec,
        )

        self._serialize_operator(
            node,
            tosa_graph,
            ts.TosaOp.Op().SELECT,
            [inputs[0].name, inputs[1].name, inputs[2].name],
            [output.name],
            None,
        )

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        import serializer.tosa_serializer as ts

        bi_supported_dtypes = [
            ts.DType.INT8,
            ts.DType.INT16,
            ts.DType.INT32,
            ts.DType.BOOL,
        ]
        self._add_node_to_tosa_graph(
            node, tosa_graph, inputs, output, bi_supported_dtypes
        )


@register_node_visitor
class WhereVisitor_FP(WhereVisitor_INT):

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
    ]

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        import serializer.tosa_serializer as ts

        mi_supported_dtypes = [
            ts.DType.FP16,
            ts.DType.FP32,
            ts.DType.INT8,
            ts.DType.INT16,
            ts.DType.INT32,
            ts.DType.BOOL,
        ]
        self._add_node_to_tosa_graph(
            node, tosa_graph, inputs, output, mi_supported_dtypes
        )
