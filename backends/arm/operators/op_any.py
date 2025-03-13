# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import cast, List

import serializer.tosa_serializer as ts  # type: ignore
from executorch.backends.arm.operators.node_visitor import (  # type: ignore
    NodeVisitor,
    register_node_visitor,
)

from executorch.backends.arm.tosa_mapping import TosaArg  # type: ignore
from serializer.tosa_serializer import TosaOp
from torch.fx import Node


@register_node_visitor
class AnyVisitor(NodeVisitor):
    target = "aten.any.dim"

    def define_node(
        self,
        node: Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:

        if not (inputs[0].dtype == output.dtype):
            raise ValueError(
                "All inputs and outputs need same dtype."
                f"Got {ts.DTypeNames[inputs[0].dtype]=}, {ts.DTypeNames[output.dtype]=}."
            )
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
            TosaOp.Op().REDUCE_ANY, [inputs[0].name], [output.name], attr
        )
