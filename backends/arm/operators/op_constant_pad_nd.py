# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Any, List

import torch

from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
)
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_specification import TosaSpecification


@register_node_visitor
class ConstantPadNDVisitor_0_80(NodeVisitor):

    target = "aten.constant_pad_nd.default"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80+BI"),
        TosaSpecification.create_from_string("TOSA-0.80+MI"),
    ]

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        import tosa_tools.v0_80.serializer.tosa_serializer as ts

        if inputs[0].dtype == ts.DType.INT8:
            input_qparams = get_input_qparams(node)
            qargs = input_qparams[0]
            pad_const_qs = qargs.quantize_value(inputs[2].number).item()
            pad_const_fp = 0.0
        else:
            pad_const_fp = inputs[2].number
            pad_const_qs = 0

        rank = len(output.shape)
        # Each dim needs 2 padding values. For example, to pad the last dimension, the pad has the form
        # (padding_left, padding_right); to pad the last two dimensions, the pad has the form
        # (padding_left, padding_right, padding_top, padding_bottom), and so on. For PyTorch NCHW format, the padding
        # values are in the reverse order. So, firstly we need to reverse the input padding parameters.
        input_pad = sum(
            [
                [inputs[1].special[i], inputs[1].special[i + 1]]
                for i in range(0, len(inputs[1].special), 2)
            ][::-1],
            [],
        )
        # Then, add dummy zeros to make sure that both input_pad and output_pad has the same size.
        input_pad = [0] * (rank * 2 - len(inputs[1].special)) + input_pad
        # For PyTorch NCHW format, dim order is [0,...,rank-1]
        input_dim_order = list(range(rank))
        output_pad = [0] * rank * 2

        # Map input padding parameters into output padding parameters. TOSA is NHWC format.
        for input_dim_idx, input_dim in enumerate(input_dim_order):
            output_dim_idx = output.dim_order.index(input_dim)
            output_pad[output_dim_idx * 2 : (output_dim_idx + 1) * 2] = input_pad[
                input_dim_idx * 2 : (input_dim_idx + 1) * 2
            ]

        attr = ts.TosaSerializerAttribute()
        attr.PadAttribute(tosa_graph.builder, output_pad, pad_const_qs, pad_const_fp)

        tosa_graph.addOperator(
            ts.TosaOp.Op().PAD, [inputs[0].name], [output.name], attr
        )


@register_node_visitor
class ConstantPadNDVisitor(NodeVisitor):

    target = "aten.constant_pad_nd.default"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
    ]

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:

        import serializer.tosa_serializer as ts  # type: ignore

        if inputs[0].dtype == ts.DType.INT8:
            input_qparams = get_input_qparams(node)
            qargs = input_qparams[0]
            pad_const_val = qargs.quantize_value(inputs[2].number).item()
            pad_const_dtype = ts.DType.INT8
        else:
            pad_const_val = inputs[2].number
            pad_const_dtype = inputs[0].dtype

        rank = len(output.shape)
        # Each dim needs 2 padding values. For example, to pad the last dimension, the pad has the form
        # (padding_left, padding_right); to pad the last two dimensions, the pad has the form
        # (padding_left, padding_right, padding_top, padding_bottom), and so on. For PyTorch NCHW format, the padding
        # values are in the reverse order. So, firstly we need to reverse the input padding parameters.
        input_pad = sum(
            [
                [inputs[1].special[i], inputs[1].special[i + 1]]
                for i in range(0, len(inputs[1].special), 2)
            ][::-1],
            [],
        )
        # Then, add dummy zeros to make sure that both input_pad and output_pad has the same size.
        input_pad = [0] * (rank * 2 - len(inputs[1].special)) + input_pad
        # For PyTorch NCHW format, dim order is [0,...,rank-1]
        input_dim_order = list(range(rank))
        output_pad = [0] * rank * 2

        # Map input padding parameters into output padding parameters. TOSA is NHWC format.
        for input_dim_idx, input_dim in enumerate(input_dim_order):
            output_dim_idx = output.dim_order.index(input_dim)
            output_pad[output_dim_idx * 2 : (output_dim_idx + 1) * 2] = input_pad[
                input_dim_idx * 2 : (input_dim_idx + 1) * 2
            ]

        padding = tosa_graph.addConst(
            shape=[len(output_pad)], dtype=ts.DType.SHAPE, vals=output_pad
        )

        pad_const = tosa_graph.addConst(
            shape=[1], dtype=pad_const_dtype, vals=[pad_const_val]
        )

        tosa_graph.addOperator(
            ts.TosaOp.Op().PAD,
            [inputs[0].name, padding.name, pad_const.name],
            [output.name],
        )
