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
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_utils import get_resize_parameters, tosa_shape

from tosa_tools.v0_80.tosa.ResizeMode import ResizeMode  # type: ignore


@register_node_visitor
class UpsampleNearest2dVisitor_0_80(NodeVisitor):
    target = "aten.upsample_nearest2d.vec"

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

        assert (
            inputs[0].shape is not None and output.shape is not None
        ), "Only static shapes are supported"

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

        attr = ts.TosaSerializerAttribute()
        attr.ResizeAttribute(
            scale=[scale_n_yx[0], scale_d_yx[0], scale_n_yx[1], scale_d_yx[1]],
            offset=offset_yx.tolist(),
            border=border_yx.tolist(),
            mode=ResizeMode.NEAREST,
        )

        tosa_graph.addOperator(
            ts.TosaOp.Op().RESIZE, [inputs[0].name], [output.name], attr
        )


@register_node_visitor
class UpsampleNearest2dVisitor(NodeVisitor):
    target = "aten.upsample_nearest2d.vec"

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

        assert (
            inputs[0].shape is not None and output.shape is not None
        ), "Only static shapes are supported"

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
