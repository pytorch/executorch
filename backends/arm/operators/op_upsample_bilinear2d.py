# Copyright 2025 Arm Limited and/or its affiliates.
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
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_quant_utils import build_rescale, build_rescale_v0_80
from executorch.backends.arm.tosa_utils import get_resize_parameters, tosa_shape


@register_node_visitor
class UpsampleBilinear2dVisitor_0_80(NodeVisitor):
    target = "aten.upsample_bilinear2d.vec"
    tosa_specs = NodeVisitor.tosa_specs_0_80

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
        from tosa_tools.v0_80.tosa.ResizeMode import ResizeMode  # type: ignore

        validate_num_inputs(self.target, inputs, 4)
        validate_same_dtype(self.target, [inputs[0], output])

        if inputs[0].shape is None or output.shape is None:
            raise ValueError("Only static shapes are supported")

        input_dtype = inputs[0].dtype

        # tosa_shape output is NHWC, take HW
        input_size_yx = torch.tensor(
            tosa_shape(inputs[0].shape, inputs[0].dim_order)[1:3]
        )
        # Ignore scale and size parameters, directly use the output size as
        # we only support static shapes currently
        output_size_yx = torch.tensor(tosa_shape(output.shape, output.dim_order)[1:3])

        scale_n_yx, scale_d_yx, offset_yx, border_yx = get_resize_parameters(
            input_size_yx, output_size_yx, ResizeMode.NEAREST, align_corners=True
        )

        def in_int16_range(x):
            return torch.all(x >= -(2**15)) and torch.all(x <= 2**15 - 1)

        if not in_int16_range(scale_n_yx):
            raise ValueError("scale_n_yx is out of the int16 range")
        if not in_int16_range(scale_d_yx):
            raise ValueError("scale_d_yx is out of the int16 range")
        if not in_int16_range(border_yx):
            raise ValueError("border_yx is out of the int16 range")

        attr = ts.TosaSerializerAttribute()
        attr.ResizeAttribute(
            scale=[scale_n_yx[0], scale_d_yx[0], scale_n_yx[1], scale_d_yx[1]],
            offset=offset_yx.tolist(),
            border=border_yx.tolist(),
            mode=ResizeMode.BILINEAR,
        )

        if input_dtype == output.dtype == ts.DType.FP32:
            tosa_graph.addOperator(
                ts.TosaOp.Op().RESIZE, [inputs[0].name], [output.name], attr
            )
            return
        elif input_dtype == output.dtype == ts.DType.INT8:
            intermediate = tosa_graph.addIntermediate(
                tosa_shape(output.shape, output.dim_order), ts.DType.INT32
            )

            tosa_graph.addOperator(
                ts.TosaOp.Op().RESIZE, [inputs[0].name], [intermediate.name], attr
            )

            final_output_scale = float(1 / (scale_n_yx[0] * scale_n_yx[1]))

            build_rescale_v0_80(
                tosa_fb=tosa_graph,
                scale=[final_output_scale],
                input_node=intermediate,
                output_name=output.name,
                output_type=ts.DType.INT8,
                input_zp=0,
                output_zp=0,
                is_double_round=False,
            )
        else:
            raise ValueError(
                "Input/output dtype not in {float32, int8}: {input_dtype=} {output.dtype=}"
            )


@register_node_visitor
class UpsampleBilinear2dVisitor(NodeVisitor):

    target = "aten.upsample_bilinear2d.vec"
    tosa_specs = NodeVisitor.tosa_specs_1_00

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
        from tosa.ResizeMode import ResizeMode  # type: ignore
        from tosa.RoundingMode import RoundingMode  # type: ignore

        validate_num_inputs(self.target, inputs, 4)
        validate_same_dtype(self.target, [inputs[0], output])

        if inputs[0].shape is None or output.shape is None:
            raise ValueError("Only static shapes are supported")

        input_dtype = inputs[0].dtype

        # tosa_shape output is NHWC, take HW
        input_size_yx = torch.tensor(
            tosa_shape(inputs[0].shape, inputs[0].dim_order)[1:3]
        )
        # Ignore scale and size parameters, directly use the output size as
        # we only support static shapes currently
        output_size_yx = torch.tensor(tosa_shape(output.shape, output.dim_order)[1:3])

        scale_n_yx, scale_d_yx, offset_yx, border_yx = get_resize_parameters(
            input_size_yx, output_size_yx, ResizeMode.NEAREST, align_corners=True
        )

        def in_int16_range(x):
            return torch.all(x >= -(2**15)) and torch.all(x <= 2**15 - 1)

        assert in_int16_range(scale_n_yx)
        assert in_int16_range(scale_d_yx)
        assert in_int16_range(border_yx)

        scales = [scale_n_yx[0], scale_d_yx[0], scale_n_yx[1], scale_d_yx[1]]

        attr = ts.TosaSerializerAttribute()
        attr.ResizeAttribute(mode=ResizeMode.BILINEAR)

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
        if input_dtype == output.dtype == ts.DType.FP32:
            tosa_graph.addOperator(
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
            return
        elif input_dtype == output.dtype == ts.DType.INT8:
            intermediate = tosa_graph.addIntermediate(
                tosa_shape(output.shape, output.dim_order), ts.DType.INT32
            )
            tosa_graph.addOperator(
                ts.TosaOp.Op().RESIZE,
                [
                    inputs[0].name,
                    scales_tensor.name,
                    offset_tensor.name,
                    border_tensor.name,
                ],
                [intermediate.name],
                attr,
            )

            final_output_scale = float(1 / (scale_n_yx[0] * scale_n_yx[1]))

            build_rescale(
                tosa_fb=tosa_graph,
                scale=[final_output_scale],
                input_node=intermediate,
                output_name=output.name,
                output_type=ts.DType.INT8,
                input_zp=0,
                output_zp=0,
                rounding_mode=RoundingMode.SINGLE_ROUND,
            )
        else:
            raise ValueError(
                "Input/output dtype not in {float32, int8}: {input_dtype=} {output.dtype=}"
            )
