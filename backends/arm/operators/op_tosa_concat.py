# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, List

import tosa_serializer as ts

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    supported_data_layout_dtypes,
    validate_num_inputs,
    validate_same_dtype,
    validate_valid_dtype,
)
from executorch.backends.arm.tosa.mapping import TosaArg
from torch.fx import Node


@register_node_visitor
class CatVisitor(NodeVisitor):
    target = "tosa.CONCAT.default"

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        supported_dtypes = supported_data_layout_dtypes(
            self.tosa_spec, allow_int16_without_extension=False
        )
        validate_num_inputs(self.target, inputs, [1, 2])
        input_tosa_args = [TosaArg(arg, self.tosa_spec) for arg in inputs[0].special]
        validate_same_dtype(self.target, [*input_tosa_args, output], ts)
        validate_valid_dtype(
            self.target,
            [*input_tosa_args, output],
            supported_dtypes,
            self.tosa_spec,
        )

        dim = node.kwargs["axis"]

        attr = ts.TosaSerializerAttribute()
        attr.ConcatAttribute(dim)

        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.CONCAT,
            [tensor.name for tensor in input_tosa_args],
            [output.name],
            attr,
        )
