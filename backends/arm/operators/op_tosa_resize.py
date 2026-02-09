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
    validate_num_inputs,
    validate_same_dtype,
    validate_valid_dtype,
)
from executorch.backends.arm.tosa.mapping import TosaArg
from executorch.backends.arm.tosa.utils import get_resize_parameters


@register_node_visitor
class ResizeVisitor(NodeVisitor):
    target = "tosa.RESIZE.default"

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        validate_num_inputs(self.target, inputs, [3, 4])
        supported_input_dtypes = [ts.DType.INT8, ts.DType.FP32, ts.DType.BF16]
        if self.tosa_spec.support_extension("int16"):
            supported_input_dtypes.append(ts.DType.INT16)
        if self.tosa_spec.support_extension("bf16"):
            supported_input_dtypes.append(ts.DType.BF16)
        validate_valid_dtype(
            self.target,
            [inputs[0]],
            supported_input_dtypes,
            self.tosa_spec,
        )
        supported_output_dtypes = [ts.DType.FP32, ts.DType.BF16]
        if node.kwargs.get("resize_mode") == "bilinear":
            resize_mode = ts.ResizeMode.BILINEAR
            align_corners = bool(node.args[2])
            supported_output_dtypes.append(ts.DType.INT32)
            if self.tosa_spec.support_extension("int16"):
                supported_output_dtypes.append(ts.DType.INT48)
        else:
            resize_mode = ts.ResizeMode.NEAREST
            align_corners = False
            validate_same_dtype(self.target, [inputs[0], output], ts)
            supported_output_dtypes.append(ts.DType.INT8)
            if self.tosa_spec.support_extension("int16"):
                supported_output_dtypes.append(ts.DType.INT16)
        validate_valid_dtype(
            self.target, [output], supported_output_dtypes, self.tosa_spec
        )
        # tosa_shape output is NHWC, take HW
        input_size_yx = tuple([inputs[0].shape[dim] for dim in inputs[0].dim_order])[
            1:3
        ]
        output_size_yx = tuple([output.shape[dim] for dim in output.dim_order])[1:3]

        # Align corners shouldn't make a difference for nearest upsampling. We set to False so
        # half pixel centers are used for resize parameter logic.
        scale_n_yx, scale_d_yx, offset_yx, border_yx = get_resize_parameters(
            input_size_yx, output_size_yx, resize_mode, align_corners=align_corners
        )

        def in_int16_range(x):
            return torch.all(x >= -(2**15)) and torch.all(x <= 2**15 - 1)

        if not in_int16_range(scale_n_yx):
            raise ValueError("scale_n_yx is out of the int16 range")
        if not in_int16_range(scale_d_yx):
            raise ValueError("scale_d_yx is out of the int16 range")
        if not in_int16_range(border_yx):
            raise ValueError("border_yx is out of the int16 range")

        scale_n_vals = [int(v) for v in scale_n_yx.tolist()]
        scale_d_vals = [int(v) for v in scale_d_yx.tolist()]
        scales = [
            scale_n_vals[0],
            scale_d_vals[0],
            scale_n_vals[1],
            scale_d_vals[1],
        ]
        scales_tensor = tosa_graph.addConst(
            [len(scales)], ts.DType.SHAPE, scales, output.name + "_scales"
        )
        offset = [int(v) for v in offset_yx.tolist()]
        offset_tensor = tosa_graph.addConst(
            [len(offset)], ts.DType.SHAPE, offset, output.name + "_offset"
        )
        border = [int(v) for v in border_yx.tolist()]
        border_tensor = tosa_graph.addConst(
            [len(border)], ts.DType.SHAPE, border, output.name + "_border"
        )
        attr = ts.TosaSerializerAttribute()
        attr.ResizeAttribute(resize_mode)

        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.RESIZE,
            [
                inputs[0].name,
                scales_tensor.name,
                offset_tensor.name,
                border_tensor.name,
            ],
            [output.name],
            attr,
        )
