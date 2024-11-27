# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import serializer.tosa_serializer as ts
import torch
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import map_dtype, TosaArg
from executorch.backends.arm.tosa_specification import Tosa_0_80
from executorch.backends.arm.tosa_utils import tosa_shape
from serializer.tosa_serializer import TosaOp


@register_node_visitor
class RshiftVisitor(NodeVisitor):
    target = "aten.__rshift__.Scalar"

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
        is_quant_node: bool,
    ) -> None:
        input_shape = inputs[0].shape
        input_0_rank = len(input_shape)
        shift_expanded_shape = [1] * input_0_rank
        dtype = node.meta["val"].dtype
        attr = ts.TosaSerializerAttribute()
        cast_input = False
        cast_output = False
        round = False
        cast_type = dtype
        if isinstance(self.tosa_spec, Tosa_0_80) and self.tosa_spec.is_U55_subset:
            # U55 only supports INT32 and round == True
            # TODO MLETORCH-525 Emulate round == False with different decomposition
            if dtype != torch.int32:
                cast_input = True
                cast_output = True
                cast_type = torch.int32
            round = True
        attr.ArithmeticRightShiftAttribute(round=round)

        if cast_input:
            # input needs to be casted to INT32
            shift_input = tosa_graph.addIntermediate(
                shape=tosa_shape(input_shape, inputs[0].dim_order),
                dtype=map_dtype(cast_type),
            )
            tosa_graph.addOperator(
                TosaOp.Op().CAST,
                [inputs[0].name],
                [shift_input.name],
                None,
            )
        else:
            shift_input = inputs[0]
        if cast_output:
            # add intermediate tensor for right shift
            shift = tosa_graph.addIntermediate(
                shape=tosa_shape(input_shape, inputs[0].dim_order),
                dtype=map_dtype(cast_type),
            )
        else:
            shift = output
        # create tensor with same rank as inputs[0]
        data = torch.full(
            shift_expanded_shape, fill_value=inputs[1].number, dtype=dtype
        )
        shift_const_name = node.name + "-shift_const"
        tosa_graph.addConst(
            shift_expanded_shape,
            map_dtype(cast_type),
            data.detach().numpy(),
            shift_const_name,
        )
        # add right shift operator
        tosa_graph.addOperator(
            TosaOp.Op().ARITHMETIC_RIGHT_SHIFT,
            [shift_input.name, shift_const_name],
            [shift.name],
            attr,
        )
        if cast_output:
            # cast output to original output dtype
            tosa_graph.addOperator(
                TosaOp.Op().CAST,
                [shift.name],
                [output.name],
                None,
            )
