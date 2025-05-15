# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Any, cast, List

import executorch.backends.arm.tosa_quant_utils as tosa_quant_utils
import torch
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
)
from executorch.backends.arm.tosa_mapping import map_dtype, TosaArg
from executorch.backends.arm.tosa_quant_utils import build_rescale

from executorch.backends.arm.tosa_specification import TosaSpecification
from torch.fx import Node


@register_node_visitor
class RescaleVisitor_0_80(NodeVisitor):
    target = "_rescale.default"

    tosa_specs = NodeVisitor.tosa_specs_0_80

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        import tosa_tools.v0_80.serializer.tosa_serializer as ts  # type: ignore

        validate_num_inputs(self.target, inputs, 5)

        input_dtype = node.all_input_nodes[0].meta["val"].dtype
        output_dtype = cast(torch.dtype, node.args[1])
        scale = cast(float, node.args[2])
        input_zp = cast(int, node.args[3])
        output_zp = cast(int, node.args[4])

        if input_dtype != torch.int8 and input_zp != 0:
            raise ValueError(
                f"If input dtype is not int8, input_zp must be 0. Got input_dtype{input_dtype=}, {input_zp=}"
            )
        if output_dtype != torch.int8 and output_zp != 0:
            raise ValueError(
                f"If output dtype is not int8, output_zp must be 0. Got {output_dtype=}, {output_zp=}"
            )

        # scale32 gives higher accuracy but for a higher HW cost.
        # For now, always go for scale32.
        scale_32 = True
        scale_width = 32 if scale_32 else 16
        multiplier, shift = tosa_quant_utils.compute_multiplier_and_shift(
            [scale], scale_width
        )
        attr_rescale = ts.TosaSerializerAttribute()
        attr_rescale.RescaleAttribute(
            input_zp=input_zp,
            output_zp=output_zp,
            multiplier=multiplier,
            shift=shift,
            scale32=scale_32,
            double_round=False,
            per_channel=False,
            input_unsigned=False,
            output_unsigned=False,
        )

        tosa_graph.addOperator(
            ts.TosaOp.Op().RESCALE, [inputs[0].name], [output.name], attr_rescale
        )


@register_node_visitor
class RescaleVisitor_INT(NodeVisitor):
    target = "_rescale.default"

    tosa_specs = [TosaSpecification.create_from_string("TOSA-1.0+INT")]

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        import serializer.tosa_serializer as ts  # type: ignore
        from tosa.RoundingMode import RoundingMode  # type: ignore

        validate_num_inputs(self.target, inputs, 5)

        input_dtype = inputs[0].dtype
        output_dtype = cast(torch.dtype, node.args[1])
        scale = cast(float, node.args[2])
        input_zp = cast(int, node.args[3])
        output_zp = cast(int, node.args[4])

        if input_dtype != map_dtype(torch.int8, self.tosa_spec) and input_zp != 0:
            raise ValueError(
                f"If input dtype is not int8, input_zp must be 0. Got input_dtype{input_dtype=}, {input_zp=}"
            )
        if output_dtype != torch.int8 and output_zp != 0:
            raise ValueError(
                f"If output dtype is not int8, output_zp must be 0. Got {ts.DTypeNames[output_dtype]}, {output_zp=}"
            )

        build_rescale(
            tosa_graph,
            scale=[scale],
            input_node=inputs[0],
            output_name=output.name,
            output_type=output.dtype,
            input_zp=input_zp,
            output_zp=output_zp,
            rounding_mode=RoundingMode.SINGLE_ROUND,
            per_channel=False,
        )
