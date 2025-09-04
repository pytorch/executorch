# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import Any, List

import torch

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
from executorch.backends.arm.tosa.utils import get_resize_parameters

from tosa.ResizeMode import ResizeMode  # type: ignore


@register_node_visitor
class UpsampleNearest2dVisitor(NodeVisitor):
    target = "aten.upsample_nearest2d.vec"

    tosa_specs = NodeVisitor.tosa_specs

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
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
            [ts.DType.INT8, ts.DType.INT32, ts.DType.FP32],
            output.tosa_spec,
        )

        # tosa_shape output is NHWC, take HW
        input_size_yx = tuple([inputs[0].shape[dim] for dim in inputs[0].dim_order])[
            1:3
        ]
        output_size_yx = tuple([output.shape[dim] for dim in output.dim_order])[1:3]

        # Align corners shouldn't make a difference for nearest upsampling. We set to False so
        # half pixel centers are used for resize parameter logic.
        scale_n_yx, scale_d_yx, offset_yx, border_yx = get_resize_parameters(
            input_size_yx, output_size_yx, ResizeMode.NEAREST, align_corners=False
        )

        def in_int16_range(x):
            return torch.all(x >= -(2**15)) and torch.all(x <= 2**15 - 1)

        if not in_int16_range(scale_n_yx):
            raise ValueError("scale_n_yx is out of the int16 range")
        if not in_int16_range(scale_d_yx):
            raise ValueError("scale_d_yx is out of the int16 range")
        if not in_int16_range(border_yx):
            raise ValueError("border_yx is out of the int16 range")

        scales = [scale_n_yx[0], scale_d_yx[0], scale_n_yx[1], scale_d_yx[1]]
        scales_tensor = tosa_graph.addConst(
            [len(scales)], ts.DType.SHAPE, scales, node.name + "_scales"
        )
        offset = offset_yx.tolist()
        offset_tensor = tosa_graph.addConst(
            [len(offset)], ts.DType.SHAPE, offset, node.name + "_offset"
        )
        border = border_yx.tolist()
        border_tensor = tosa_graph.addConst(
            [len(border)], ts.DType.SHAPE, border, node.name + "_border"
        )
        attr = ts.TosaSerializerAttribute()
        attr.ResizeAttribute(
            mode=ResizeMode.NEAREST,
        )

        self._serialize_operator(
            node,
            tosa_graph,
            ts.TosaOp.Op().RESIZE,
            [
                inputs[0].name,
                scales_tensor.name,
                offset_tensor.name,
                border_tensor.name,
            ],
            [output.name],
            attr,
        )
