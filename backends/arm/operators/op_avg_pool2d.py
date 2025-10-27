# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import Any, List

import serializer.tosa_serializer as ts

import torch

from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
    get_output_qparams,
)
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    adjust_pooling_pad_if_needed,
    validate_num_inputs,
    validate_same_dtype,
    validate_valid_dtype,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.arm.tosa.mapping import TosaArg


@register_node_visitor
class AvgPool2dVisitor(NodeVisitor):
    target = "aten.avg_pool2d.default"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
    ]

    def __init__(self, *args):
        super().__init__(*args)

    def _build_generic_avgpool2d(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
        input_zp: int,
        output_zp: int,
        accumulator_type: Any,
    ) -> None:

        input_tensor = inputs[0]
        kernel_size_list = inputs[1].special
        stride_size_list = inputs[2].special

        if len(inputs) > 4:
            ceil_mode = bool(inputs[4].number)
        else:
            ceil_mode = False

        try:
            pad_size_list = inputs[3].special
            pad_size_list = [
                pad_size_list[0],
                pad_size_list[0],
                pad_size_list[1],
                pad_size_list[1],
            ]
        except IndexError:
            pad_size_list = [0, 0, 0, 0]

        # Adjust the padding as necessary
        pad_size_list[1] = adjust_pooling_pad_if_needed(
            input_tensor.shape[2],
            kernel_size_list[0],
            stride_size_list[0],
            pad_size_list[1],
            ceil_mode,
        )
        pad_size_list[3] = adjust_pooling_pad_if_needed(
            input_tensor.shape[3],
            kernel_size_list[1],
            stride_size_list[1],
            pad_size_list[3],
            ceil_mode,
        )

        attr = ts.TosaSerializerAttribute()
        attr.AvgPool2dAttribute(
            kernel=kernel_size_list,
            stride=stride_size_list,
            pad=pad_size_list,
            acc_type=accumulator_type,
        )
        input_zp_tensor = tosa_graph.addConst(
            shape=[1], dtype=output.dtype, vals=[input_zp]
        )
        output_zp_tensor = tosa_graph.addConst(
            shape=[1], dtype=output.dtype, vals=[output_zp]
        )

        self._serialize_operator(
            node,
            tosa_graph,
            ts.TosaOp.Op().AVG_POOL2D,
            [input_tensor.name, input_zp_tensor.name, output_zp_tensor.name],
            [output.name],
            attr,
        )

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        validate_num_inputs(self.target, inputs, [3, 4, 5, 6, 7])
        validate_same_dtype(self.target, [inputs[0], output], ts)
        validate_valid_dtype(
            self.target, [inputs[0], output], ts.DType.INT8, output.tosa_spec
        )

        accumulator_type = ts.DType.INT32

        input_qargs = get_input_qparams(node)
        input_zp = input_qargs[0].get_zp_per_tensor()

        output_qargs = get_output_qparams(node)
        output_zp = output_qargs[0].get_zp_per_tensor()

        self._build_generic_avgpool2d(
            node, tosa_graph, inputs, output, input_zp, output_zp, accumulator_type
        )


@register_node_visitor
class AvgPool2dVisitor_FP(AvgPool2dVisitor):
    target = "aten.avg_pool2d.default"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
    ]

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        validate_num_inputs(self.target, inputs, [3, 4, 5, 6, 7])
        validate_same_dtype(self.target, [inputs[0], output], ts)
        validate_valid_dtype(
            self.target,
            [inputs[0], output],
            [ts.DType.INT8, ts.DType.FP32],
            output.tosa_spec,
        )

        if inputs[0].dtype == ts.DType.INT8:
            super().define_node(node, tosa_graph, inputs, output)

        if inputs[0].dtype == ts.DType.FP32:
            accumulator_type = ts.DType.FP32
            # Initilize zero point to zero.
            input_zp = 0
            output_zp = 0

            self._build_generic_avgpool2d(
                node, tosa_graph, inputs, output, input_zp, output_zp, accumulator_type
            )
