# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import cast, List

import executorch.backends.arm.tosa_quant_utils as tqutils
import executorch.backends.arm.tosa_utils as tutils

import serializer.tosa_serializer as ts
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_specification import TosaSpecification
from serializer.tosa_serializer import TosaOp
from torch.fx import Node


@register_node_visitor
class SumVisitor_080_BI(NodeVisitor):
    target = "aten.sum.dim_IntList"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80+BI"),
    ]

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        input_shape = list(inputs[0].shape)
        dim_list = cast(list[int], inputs[1].special)
        dim_list = [dim % len(input_shape) for dim in dim_list]
        keep_dim = cast(bool, inputs[2].number if len(inputs) > 2 else False)
        assert keep_dim, "This case should be handled by InsertSqueezeAfterSumPass"

        # Rescale input to 32 bit
        rescaled_inputs, scale = tqutils.insert_rescale_ops_to_int32(
            tosa_graph,
            [inputs[0]],
            node,
        )

        prev_node = rescaled_inputs[0]
        reduced_shape = input_shape

        # Reduce all dims in dim_list one-by-one.
        for dim in dim_list:
            # When reduced, the size of the dim becomes 1.
            reduced_shape[dim] = 1

            attr = ts.TosaSerializerAttribute()
            attr.AxisAttribute(inputs[0].dim_order.index(dim))

            next_node = tosa_graph.addIntermediate(
                tutils.tosa_shape(reduced_shape, inputs[0].dim_order),
                dtype=ts.DType.INT32,
            )

            tosa_graph.addOperator(
                TosaOp.Op().REDUCE_SUM, [prev_node.name], [next_node.name], attr
            )

            prev_node = next_node
        tqutils.insert_rescale_op_to_int8(tosa_graph, prev_node, scale, node)


@register_node_visitor
class SumVisitor_080_MI(SumVisitor_080_BI):
    # inheriting 'target' from BI class

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80+MI"),
    ]

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        if inputs[0].dtype == ts.DType.INT8:
            return super().define_node(node, tosa_graph, inputs, output)
        input_name = inputs[0].name
        reduced_shape = list(inputs[0].shape)
        dim_list = cast(list[int], inputs[1].special)
        dim_list = [dim % len(reduced_shape) for dim in dim_list]
        keep_dim = cast(bool, inputs[2].number if len(inputs) > 2 else False)
        assert keep_dim, "This case should be handled by InsertSqueezeAfterSumPass"

        # Reduce all dims in dim_list one-by-one.
        for dim in dim_list:
            # When reduced, the size of the dim becomes 1
            reduced_shape[dim] = 1

            attr = ts.TosaSerializerAttribute()
            attr.AxisAttribute(inputs[0].dim_order.index(dim))

            if dim == dim_list[-1]:
                output_name = output.name
            else:
                output_name = tosa_graph.addIntermediate(
                    tutils.tosa_shape(reduced_shape, inputs[0].dim_order),
                    dtype=ts.DType.FP32,
                ).name

            tosa_graph.addOperator(
                TosaOp.Op().REDUCE_SUM, [input_name], [output_name], attr
            )

            input_name = output_name
