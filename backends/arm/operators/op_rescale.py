# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import cast, List

import executorch.backends.arm.tosa_quant_utils as tosa_quant_utils
import serializer.tosa_serializer as ts  # type: ignore
import torch

import tosa.Op as TosaOp  # type: ignore
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import map_dtype, TosaArg
from torch.fx import Node


@register_node_visitor
class RescaleVisitor(NodeVisitor):
    target = "_rescale.default"

    def define_node(
        self,
        node: Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:

        input_dtype = inputs[0].dtype
        output_dtype = cast(torch.dtype, node.args[1])
        scale = cast(float, node.args[2])
        input_zp = cast(int, node.args[3])
        output_zp = cast(int, node.args[4])

        # Skip int16 cases for now.
        if input_dtype != map_dtype(torch.int8) and input_zp != 0:
            raise ValueError(
                f"If input dtype is not int8, input_zp must be 0. Got input_dtype{ts.DTypeNames[input_dtype]}, {input_zp=}"
            )
        if output_dtype != torch.int8 and output_zp != 0:
            raise ValueError(
                f"If output dtype is not int8, output_zp must be 0. Got {output_dtype=}, {output_zp=}"
            )

        scale_width = 32 if output_dtype == torch.int32 else 16
        multiplier, shift = tosa_quant_utils.compute_multiplier_and_shift(
            scale, scale_width
        )
        attr_rescale = ts.TosaSerializerAttribute()
        attr_rescale.RescaleAttribute(
            input_zp=input_zp,
            output_zp=output_zp,
            multiplier=[multiplier],
            shift=[shift],
            scale32=output_dtype == torch.int32,
            double_round=False,
            per_channel=False,
            input_unsigned=False,
            output_unsigned=False,
        )

        tosa_graph.addOperator(
            TosaOp.Op().RESCALE, [inputs[0].name], [output.name], attr_rescale
        )
