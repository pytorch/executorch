# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
"""Provide a visitor for lowering batched matmul (BMM) to TOSA."""

from typing import Any, List

import torch

from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
    get_output_qparams,
)
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_same_dtype,
    validate_valid_dtype,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.arm.tosa.mapping import TosaArg
from executorch.backends.arm.tosa.quant_utils import build_rescale
from tosa.RoundingMode import RoundingMode  # type: ignore


@register_node_visitor
class BMMVisitor(NodeVisitor):
    """Provide a visitor that lowers ``aten.bmm`` to TOSA ``MATMUL``.

    INT8 accumulates into INT32; add a rescale to INT8 using SINGLE_ROUND
    rounding and output zero-point.

    """

    target = "aten.bmm.default"

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
        """Define the TOSA ``MATMUL`` operator and optional rescale."""
        import serializer.tosa_serializer as ts  # type: ignore

        validate_num_inputs(self.target, inputs, 2)
        validate_same_dtype(self.target, [*inputs, output], ts)
        validate_valid_dtype(
            self.target,
            [*inputs, output],
            [ts.DType.INT8, ts.DType.INT16, ts.DType.FP32],
            output.tosa_spec,
        )

        # aten.bmm maps directly to MATMUL

        # For INT8, we need to get the zero points and add an intermediate tensor
        # for a later rescale.

        if inputs[0].dtype == ts.DType.INT8:
            input_qparams = get_input_qparams(node)
            input0_zp = input_qparams[0].get_zp_per_tensor()
            input1_zp = input_qparams[1].get_zp_per_tensor()
            bmm_result = tosa_graph.addIntermediate(output.shape, ts.DType.INT32)
            bmm_output_name = bmm_result.name
        else:
            bmm_output_name = output.name
            input0_zp, input1_zp = 0, 0

        tosa_graph.addConst([1], inputs[0].dtype, [input0_zp], name=f"{node.name}_A_ZP")
        tosa_graph.addConst([1], inputs[1].dtype, [input1_zp], name=f"{node.name}_B_ZP")

        # Add the MATMUL to the TOSA graph.
        self._serialize_operator(
            node,
            tosa_graph,
            ts.TosaOp.Op().MATMUL,
            [
                inputs[0].name,
                inputs[1].name,
                f"{node.name}_A_ZP",
                f"{node.name}_B_ZP",
            ],
            [bmm_output_name],
        )

        # As INT8 accumulates into INT32, we need to rescale it back to INT8
        if output.dtype == ts.DType.INT8:
            output_qparams = get_output_qparams(node)[0]
            final_output_scale = (
                input_qparams[0].get_scale_per_tensor() * input_qparams[1].get_scale_per_tensor()  # type: ignore[possibly-undefined]  # pyre-ignore[61]
            ) / output_qparams.get_scale_per_tensor()

            build_rescale(
                tosa_fb=tosa_graph,
                scale=[final_output_scale],
                # pyre-ignore[61]: Uninitialized local [61]: Local variable `bmm_result` is undefined, or not always defined.
                input_node=bmm_result,  # type: ignore[possibly-undefined]
                output_name=output.name,
                output_type=ts.DType.INT8,
                input_zp=[0],
                output_zp=[output_qparams.get_zp_per_tensor()],
                rounding_mode=RoundingMode.SINGLE_ROUND,
            )
