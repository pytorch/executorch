# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
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

from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_quant_utils import build_rescale, build_rescale_v0_80
from executorch.backends.arm.tosa_specification import TosaSpecification
from tosa.RoundingMode import RoundingMode  # type: ignore


@register_node_visitor
class BMMVisitor_0_80(NodeVisitor):
    target = "aten.bmm.default"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80+BI"),
        TosaSpecification.create_from_string("TOSA-0.80+MI"),
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

        import tosa_tools.v0_80.serializer.tosa_serializer as ts  # type: ignore

        if inputs[0].dtype != inputs[1].dtype or inputs[0].dtype != output.dtype:
            raise TypeError(
                f"All IO needs to have the same data type, got: "
                f"{inputs[0].dtype=}, {inputs[1].dtype=} and {output.dtype=}"
            )

        # aten.bmm maps directly to MATMUL
        # NOTE: For now, only INT8 & FP32 is supported
        supported_dtypes = [ts.DType.INT8, ts.DType.FP32]
        for input in inputs:
            if input.dtype not in supported_dtypes:
                raise TypeError(
                    f'IO data type needs to be {supported_dtypes}, got "{input.dtype}"'
                )

        # aten.bmm maps directly to MATMUL

        # For INT8, we need to get the zero points and add an intermediate tensor
        # for a later rescale.
        if inputs[0].dtype == ts.DType.INT8:
            input_qparams = get_input_qparams(node)
            input0_zp = input_qparams[0].zp
            input1_zp = input_qparams[1].zp
            bmm_result = tosa_graph.addIntermediate(output.shape, ts.DType.INT32)
            bmm_output_name = bmm_result.name
        else:
            bmm_output_name = output.name
            input0_zp, input1_zp = 0, 0

        # Add the MATMUL to the TOSA graph.
        attr = ts.TosaSerializerAttribute()
        attr.MatMulAttribute(A_zp=input0_zp, B_zp=input1_zp)

        tosa_graph.addOperator(
            ts.TosaOp.Op().MATMUL,
            [inputs[0].name, inputs[1].name],
            [bmm_output_name],
            attr,
        )

        # As INT8 accumulates into INT32, we need to rescale it back to INT8
        if output.dtype == ts.DType.INT8:
            output_qparams = get_output_qparams(node)[0]
            final_output_scale = (
                input_qparams[0].scale * input_qparams[1].scale  # type: ignore[possibly-undefined]  # pyre-ignore[61]
            ) / output_qparams.scale

            build_rescale_v0_80(
                tosa_fb=tosa_graph,
                scale=[final_output_scale],
                # pyre-ignore[61]: Uninitialized local [61]: Local variable `bmm_result` is undefined, or not always defined.
                input_node=bmm_result,  # type: ignore[possibly-undefined]
                output_name=output.name,
                output_type=ts.DType.INT8,
                input_zp=0,
                output_zp=output_qparams.zp,
                is_double_round=False,
            )


@register_node_visitor
class BMMVisitor(NodeVisitor):
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

        import serializer.tosa_serializer as ts  # type: ignore

        if inputs[0].dtype != inputs[1].dtype or inputs[0].dtype != output.dtype:
            raise TypeError(
                f"All IO needs to have the same data type, got: "
                f"{inputs[0].dtype=}, {inputs[1].dtype=} and {output.dtype=}"
            )

        # aten.bmm maps directly to MATMUL
        # NOTE: For now, only INT8 & FP32 is supported
        supported_dtypes = [ts.DType.INT8, ts.DType.FP32]
        for input in inputs:
            if input.dtype not in supported_dtypes:
                raise TypeError(
                    f'IO data type needs to be {supported_dtypes}, got "{input.dtype}"'
                )

        # aten.bmm maps directly to MATMUL
        # NOTE: For now, only INT8 & FP32 is supported

        # For INT8, we need to get the zero points and add an intermediate tensor
        # for a later rescale.

        if inputs[0].dtype == ts.DType.INT8:
            input_qparams = get_input_qparams(node)
            input0_zp = input_qparams[0].zp
            input1_zp = input_qparams[1].zp
            bmm_result = tosa_graph.addIntermediate(output.shape, ts.DType.INT32)
            bmm_output_name = bmm_result.name
        else:
            bmm_output_name = output.name
            input0_zp, input1_zp = 0, 0

        tosa_graph.addConst([1], inputs[0].dtype, [input0_zp], name=f"{node.name}_A_ZP")
        tosa_graph.addConst([1], inputs[1].dtype, [input1_zp], name=f"{node.name}_B_ZP")

        # Add the MATMUL to the TOSA graph.
        tosa_graph.addOperator(
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
                input_qparams[0].scale * input_qparams[1].scale  # type: ignore[possibly-undefined]  # pyre-ignore[61]
            ) / output_qparams.scale

            build_rescale(
                tosa_fb=tosa_graph,
                scale=[final_output_scale],
                # pyre-ignore[61]: Uninitialized local [61]: Local variable `bmm_result` is undefined, or not always defined.
                input_node=bmm_result,  # type: ignore[possibly-undefined]
                output_name=output.name,
                output_type=ts.DType.INT8,
                input_zp=0,
                output_zp=output_qparams.zp,
                rounding_mode=RoundingMode.SINGLE_ROUND,
            )
