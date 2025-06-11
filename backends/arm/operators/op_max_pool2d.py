# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import Any, List

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
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_specification import TosaSpecification


@register_node_visitor
class MaxPool2dVisitor_0_80(NodeVisitor):
    target = "aten.max_pool2d.default"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80+BI"),
        TosaSpecification.create_from_string("TOSA-0.80+MI"),
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
        import tosa_tools.v0_80.serializer.tosa_serializer as ts  # type: ignore

        validate_num_inputs(self.target, inputs, [3, 4])
        validate_same_dtype(self.target, [inputs[0], output], ts)

        input_tensor = inputs[0]
        kernel_size = inputs[1].special
        stride = inputs[2].special

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
            kernel_size[0],
            stride[0],
            pad_size_list[1],
        )
        pad_size_list[3] = adjust_pooling_pad_if_needed(
            input_tensor.shape[3],
            kernel_size[1],
            stride[1],
            pad_size_list[3],
        )

        accumulator_type = output.dtype

        # Initilize zero point to zero.
        input_zp = 0
        if inputs[0].dtype == ts.DType.INT8:
            input_qparams = get_input_qparams(node)
            input_zp = input_qparams[0].zp

        output_zp = 0
        if output.dtype == ts.DType.INT8:
            output_qparams = get_output_qparams(node)
            output_zp = output_qparams[0].zp

        attr = ts.TosaSerializerAttribute()
        attr.PoolAttribute(
            kernel=kernel_size,
            stride=stride,
            pad=pad_size_list,
            input_zp=input_zp,
            output_zp=output_zp,
            accum_dtype=accumulator_type,
        )

        tosa_graph.addOperator(
            ts.TosaOp.Op().MAX_POOL2D,
            [input_tensor.name],
            [output.name],
            attr,
        )


@register_node_visitor
class MaxPool2dVisitor(NodeVisitor):
    target = "aten.max_pool2d.default"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
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

        import serializer.tosa_serializer as ts  # type: ignore

        validate_num_inputs(self.target, inputs, [3, 4])
        validate_same_dtype(self.target, [inputs[0], output], ts)

        input_tensor = inputs[0]
        kernel_size = inputs[1].special
        stride = inputs[2].special

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
            kernel_size[0],
            stride[0],
            pad_size_list[1],
        )
        pad_size_list[3] = adjust_pooling_pad_if_needed(
            input_tensor.shape[3],
            kernel_size[1],
            stride[1],
            pad_size_list[3],
        )

        attr = ts.TosaSerializerAttribute()
        attr.MaxPool2dAttribute(
            kernel=kernel_size, stride=stride, pad=pad_size_list, nan_mode=1
        )

        tosa_graph.addOperator(
            ts.TosaOp.Op().MAX_POOL2D,
            [input_tensor.name],
            [output.name],
            attr,
        )
