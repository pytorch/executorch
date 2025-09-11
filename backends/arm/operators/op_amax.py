# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, List

from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
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
from torch.fx import Node


@register_node_visitor
class MaxVisitor(NodeVisitor):
    target = "aten.amax.default"

    tosa_specs = NodeVisitor.tosa_specs

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

        validate_num_inputs(self.target, inputs, 3)
        validate_same_dtype(self.target, [inputs[0], output], ts)
        validate_valid_dtype(
            self.target,
            [inputs[0], output],
            [ts.DType.INT8, ts.DType.INT16, ts.DType.INT32, ts.DType.FP32],
            output.tosa_spec,
        )

        input = inputs[0]
        dim = inputs[1].number

        if dim < 0:
            tensor = get_first_fake_tensor(node)
            rank = len(tensor.size())
            dim = rank + dim

        keep_dims = inputs[2].number
        if not keep_dims:
            raise RuntimeError(
                "TOSA only supports keepdims == True; Did you run the convert_minmax pass?"
            )

        attr = ts.TosaSerializerAttribute()
        attr.ReduceMaxAttribute(axis=input.dim_order.index(dim), nan_mode=1)
        self._serialize_operator(
            node,
            tosa_graph,
            ts.TosaOp.Op().REDUCE_MAX,
            [input.name],
            [output.name],
            attr,
        )
