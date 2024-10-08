# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import List

import serializer.tosa_serializer as ts
import torch
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_utils import tosa_shape
from serializer.tosa_serializer import TosaOp


@register_node_visitor
class SoftmaxVisitor(NodeVisitor):
    target = "aten._softmax.default"

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
        is_quant_node: bool,
    ) -> None:
        input_name = inputs[0].name
        dim_order = inputs[0].dim_order
        input_shape = tosa_shape(inputs[0].shape, dim_order)
        dim_value = dim_order.index(inputs[1].number % len(dim_order))

        ## softmax = exp(logits - max(logits)) / reduce_sum(exp(logits - max(logits)), -1)
        # FP32
        # reduce_max_res = reducemax(logits)
        # sub_res = sub(inputs, reduce_max_res)
        # exp_res = exp(sub_res)
        # reduce_sum_res = reduce_sum(exp_res, -1)
        # inverted_reduce_sum = reciprocal(reduce_sum_res)
        # output = mul(exp_res, inverted_reduce_sum)

        # Max_Reduction
        attr_axis = ts.TosaSerializerAttribute()
        attr_axis.AxisAttribute(axis=dim_value)
        reduced_shape = list(input_shape)
        reduced_shape[dim_value] = 1
        reduce_max_res = tosa_graph.addIntermediate(reduced_shape, output.dtype)
        tosa_graph.addOperator(
            TosaOp.Op().REDUCE_MAX,
            [input_name],
            [reduce_max_res.name],
            attr_axis,
        )

        # Subtract max from logits
        sub_res = tosa_graph.addIntermediate(input_shape, output.dtype)
        tosa_graph.addOperator(
            TosaOp.Op().SUB,
            [input_name, reduce_max_res.name],
            [sub_res.name],
        )

        # Raise the subtraction results to exponent
        exp_res = tosa_graph.addIntermediate(input_shape, output.dtype)
        tosa_graph.addOperator(TosaOp.Op().EXP, [sub_res.name], [exp_res.name])

        # Reduce_sum of the calculated exponent value
        reduce_sum_res = tosa_graph.addIntermediate(reduced_shape, output.dtype)
        tosa_graph.addOperator(
            TosaOp.Op().REDUCE_SUM,
            [exp_res.name],
            [reduce_sum_res.name],
            attr_axis,
        )

        # Invert the reduce_sum
        inverted_reduce_sum = tosa_graph.addIntermediate(reduced_shape, output.dtype)
        tosa_graph.addOperator(
            TosaOp.Op().RECIPROCAL,
            [reduce_sum_res.name],
            [inverted_reduce_sum.name],
        )

        # Multiply two parts to get the final results
        attr_mul = ts.TosaSerializerAttribute()
        attr_mul.MulAttribute(0)
        tosa_graph.addOperator(
            TosaOp.Op().MUL,
            [exp_res.name, inverted_reduce_sum.name],
            [output.name],
            attr_mul,
        )
