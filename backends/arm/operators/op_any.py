# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import Any, cast, List

from executorch.backends.arm.operators.node_visitor import (  # type: ignore
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_same_dtype,
)

from executorch.backends.arm.tosa_mapping import TosaArg  # type: ignore
from torch.fx import Node


@register_node_visitor
class AnyVisitor_0_80(NodeVisitor):
    target = "aten.any.dim"

    tosa_specs = NodeVisitor.tosa_specs_0_80

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        import tosa_tools.v0_80.serializer.tosa_serializer as ts  # type: ignore

        validate_num_inputs(self.target, inputs, 3)
        validate_same_dtype(self.target, [inputs[0], output], ts)

        if not (inputs[0].dtype == ts.DType.BOOL):
            raise ValueError("All inputs need to be BOOL." f"Got {inputs[0].dtype=}")

        input_shape = list(inputs[0].shape)
        dim = cast(int, inputs[1].number) % len(
            input_shape
        )  # process the negative index
        keep_dim = cast(bool, inputs[2].number if len(inputs) > 2 else False)
        if not keep_dim:
            raise ValueError("This case should be handled by ConvertAnyDimDimsPass")

        attr = ts.TosaSerializerAttribute()
        attr.AxisAttribute(inputs[0].dim_order.index(dim))

        tosa_graph.addOperator(
            ts.TosaOp.Op().REDUCE_ANY, [inputs[0].name], [output.name], attr
        )


@register_node_visitor
class AnyVisitor(NodeVisitor):
    target = "aten.any.dim"

    tosa_specs = NodeVisitor.tosa_specs_1_00

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

        if not (inputs[0].dtype == ts.DType.BOOL):
            raise ValueError("All inputs need to be BOOL." f"Got {inputs[0].dtype=}")

        input_shape = list(inputs[0].shape)
        dim = cast(int, inputs[1].number) % len(
            input_shape
        )  # process the negative index
        keep_dim = cast(bool, inputs[2].number if len(inputs) > 2 else False)
        if not keep_dim:
            raise ValueError("This case should be handled by ConvertAnyDimDimsPass")

        attr = ts.TosaSerializerAttribute()
        attr.ReduceAnyAttribute(inputs[0].dim_order.index(dim))

        tosa_graph.addOperator(
            ts.TosaOp.Op().REDUCE_ANY, [inputs[0].name], [output.name], attr
        )
