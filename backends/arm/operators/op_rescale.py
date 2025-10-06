# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Any, cast, List

import torch
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
)

from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.arm.tosa.mapping import map_dtype, TosaArg
from executorch.backends.arm.tosa.quant_utils import build_rescale
from torch.fx import Node


@register_node_visitor
class RescaleVisitor(NodeVisitor):
    target = "tosa.RESCALE.default"

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

        if (
            input_dtype
            not in [
                map_dtype(torch.int8, self.tosa_spec),
                map_dtype(torch.int16, self.tosa_spec),
            ]
            and input_zp != 0
        ):
            raise ValueError(
                f"If input dtype is not int8 or int16, input_zp must be 0. Got input_dtype{input_dtype=}, {input_zp=}"
            )
        if output_dtype not in [torch.int8, torch.int16] and output_zp != 0:
            raise ValueError(
                f"If output dtype is not int8 or int16, output_zp must be 0. Got {ts.DTypeNames[output_dtype]}, {output_zp=}"
            )

        build_rescale(
            tosa_graph,
            scale=[scale],
            input_node=inputs[0],
            output_name=output.name,
            output_type=output.dtype,
            input_zp=[input_zp],
            output_zp=[output_zp],
            rounding_mode=RoundingMode.SINGLE_ROUND,
            per_channel=False,
        )
