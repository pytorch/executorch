# Copyright 2024 Arm Limited and/or its affiliates.
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
from executorch.backends.arm.tosa_utils import get_resize_parameters, tosa_shape
from serializer.tosa_serializer import TosaOp

from tosa.ResizeMode import ResizeMode


@register_node_visitor
class UpsampleNearest2dVisitor(NodeVisitor):
    target = "aten.upsample_nearest2d.vec"

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
            TosaOp.Op().RESIZE, [inputs[0].name], [output.name], attr
        )
