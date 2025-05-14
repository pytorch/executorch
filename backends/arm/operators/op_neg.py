# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import Any, List

import torch.fx

from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
    get_output_qparams,
)
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)

from executorch.backends.arm.tosa_mapping import TosaArg


def get_negate_zero_points(node: torch.fx.Node, is_int8: bool) -> tuple[int, int]:
    """
    Returns (input1_zp, output_zp) for TOSA NEGATE.
    Must be zero for non-int8 types.
    """
    if is_int8:
        return (
            get_input_qparams(node)[0].zp,
            get_output_qparams(node)[0].zp,
        )
    return (0, 0)


@register_node_visitor
class NegVisitor_0_80(NodeVisitor):
    target = "aten.neg.default"

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

        supported_dtypes = {
            ts.DType.INT8,
            ts.DType.INT16,
            ts.DType.INT32,
            ts.DType.FP16,
            ts.DType.BF16,
            ts.DType.FP32,
        }

        if inputs[0].dtype not in supported_dtypes:
            raise ValueError(f"Unsupported dtype for NEGATE: {inputs[0].dtype}")

        if inputs[0].dtype != output.dtype:
            raise ValueError(
                "All inputs and output need same dtype."
                f"Got {inputs[0].dtype=}, {output.dtype=}"
            )
        input_zp, output_zp = get_negate_zero_points(
            node, inputs[0].dtype == ts.DType.INT8
        )

        attr = ts.TosaSerializerAttribute()
        attr.NegateAttribute(input1_zp=input_zp, output_zp=output_zp)
        tosa_graph.addOperator(
            ts.TosaOp.Op().NEGATE,
            [inputs[0].name],
            [output.name],
            attributes=attr,
        )


@register_node_visitor
class NegVisitor(NodeVisitor):
    target = "aten.neg.default"

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
        import serializer.tosa_serializer as ts  # type: ignore

        supported_dtypes = {
            ts.DType.INT8,
            ts.DType.INT16,
            ts.DType.INT32,
            ts.DType.FP16,
            ts.DType.BF16,
            ts.DType.FP32,
        }

        if inputs[0].dtype not in supported_dtypes:
            raise ValueError(f"Unsupported dtype for NEGATE: {inputs[0].dtype}")

        if inputs[0].dtype != output.dtype:
            raise ValueError(
                "All inputs and output need same dtype."
                f"Got {inputs[0].dtype=}, {output.dtype=}"
            )
        input_zp, output_zp = get_negate_zero_points(
            node, inputs[0].dtype == ts.DType.INT8
        )

        input_zp_tensor = tosa_graph.addConst(
            (1,), inputs[0].dtype, [input_zp], name=output.name + "_input_zp"
        )

        output_zp_tensor = tosa_graph.addConst(
            (1,), output.dtype, [output_zp], name=output.name + "_output_zp"
        )

        tosa_graph.addOperator(
            ts.TosaOp.Op().NEGATE,
            [inputs[0].name, input_zp_tensor.name, output_zp_tensor.name],
            [output.name],
        )
