# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Provide a visitor for lowering block-scaled matmul to TOSA."""

from typing import Any, List

import torch
import tosa_serializer as ts

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_valid_dtype,
)
from executorch.backends.arm.tosa.mapping import TosaArg
from executorch.backends.arm.tosa.specification import TosaSpecification


@register_node_visitor
class MatMulTBlockScaledVisitor(NodeVisitor):
    """Serialize TOSA ``MATMUL_T_BLOCK_SCALED``."""

    target = "tosa.MATMUL_T_BLOCK_SCALED.default"
    tosa_specs = [TosaSpecification.create_from_string("TOSA-1.1+FP")]

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        # The tosa_specs attribute cannot express extension requirements.
        # Therefore, check for the extension explicitly here.
        if not self.tosa_spec.support_extension("mxfp"):
            raise ValueError(f"{self.target} requires the TOSA mxfp extension")

        validate_num_inputs(self.target, inputs, 5)

        (
            A_data,
            A_scale,
            B_data,
            B_scale,
        ) = inputs[:4]
        block_size = inputs[4].number

        validate_valid_dtype(
            self.target,
            [A_data, B_data],
            [
                ts.DType.FP4E2M1,
                ts.DType.FP6E2M3,
                ts.DType.FP6E3M2,
                ts.DType.FP8E4M3,
                ts.DType.FP8E5M2,
            ],
            self.tosa_spec,
        )
        validate_valid_dtype(
            self.target,
            [A_scale, B_scale],
            ts.DType.FP8UE8M0,
            self.tosa_spec,
        )
        validate_valid_dtype(
            self.target,
            output,
            ts.DType.FP32,
            self.tosa_spec,
        )
        if block_size != 32:
            raise ValueError(f"Invalid block size {block_size}")

        if A_data.dtype != B_data.dtype:
            raise ValueError(
                f"{self.target}: payload dtypes must match, got {inputs[0].dtype} and {inputs[2].dtype}"
            )

        attr = ts.TosaSerializerAttribute()
        attr.MatMulTBlockScaledAttribute(block_size)

        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.MATMUL_T_BLOCK_SCALED,
            [
                inputs[0].name,
                inputs[1].name,
                inputs[2].name,
                inputs[3].name,
            ],
            [output.name],
            attr,
        )
