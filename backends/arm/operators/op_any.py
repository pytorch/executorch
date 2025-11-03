# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast, List

import tosa_serializer as ts

from executorch.backends.arm.operators.node_visitor import (  # type: ignore
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_same_dtype,
    validate_valid_dtype,
)

from executorch.backends.arm.tosa.mapping import TosaArg  # type: ignore
from torch.fx import Node


@register_node_visitor
class AnyVisitor(NodeVisitor):
    target = "aten.any.dim"

    tosa_specs = NodeVisitor.tosa_specs

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        validate_num_inputs(self.target, inputs, 3)
        validate_same_dtype(self.target, [inputs[0], output], ts)
        validate_valid_dtype(
            self.target, [inputs[0], output], ts.DType.BOOL, output.tosa_spec
        )

        input_shape = list(inputs[0].shape)
        dim = cast(int, inputs[1].number) % len(
            input_shape
        )  # process the negative index
        keep_dim = cast(bool, inputs[2].number if len(inputs) > 2 else False)
        if not keep_dim:
            raise ValueError("This case should be handled by ConvertAnyDimDimsPass")

        attr = ts.TosaSerializerAttribute()
        attr.ReduceAnyAttribute(inputs[0].dim_order.index(dim))

        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.REDUCE_ANY,
            [inputs[0].name],
            [output.name],
            attr,
        )
