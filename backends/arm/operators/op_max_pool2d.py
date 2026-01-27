# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List

import torch

import tosa_serializer as ts

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
        validate_num_inputs(self.target, inputs, [3, 4, 5, 6])
        validate_same_dtype(self.target, [inputs[0], output], ts)
        supported_dtypes = [ts.DType.INT8, ts.DType.FP32]
        if self.tosa_spec.support_extension("int16"):
            supported_dtypes.append(ts.DType.INT16)
        validate_valid_dtype(
            self.target,
            [inputs[0], output],
            supported_dtypes,
            self.tosa_spec,
        )

        input_tensor = inputs[0]
        kernel_size = inputs[1].special
        stride = inputs[2].special

        if len(inputs) == 6:
            ceil_mode = bool(inputs[5].number)
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
        except (IndexError, AttributeError):
            pad_size_list = [0, 0, 0, 0]

        # Adjust the padding as necessary
        pad_size_list[1] = adjust_pooling_pad_if_needed(
            input_tensor.shape[2],
            kernel_size[0],
            stride[0],
            pad_size_list[1],
            ceil_mode,
        )
        pad_size_list[3] = adjust_pooling_pad_if_needed(
            input_tensor.shape[3],
            kernel_size[1],
            stride[1],
            pad_size_list[3],
            ceil_mode,
        )

        attr = ts.TosaSerializerAttribute()
        attr.MaxPool2dAttribute(
            kernel=kernel_size,
            stride=stride,
            pad=pad_size_list,
            nan_mode=ts.NanPropagationMode.PROPAGATE,
        )

        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.MAX_POOL2D,
            [input_tensor.name],
            [output.name],
            attr,
        )
