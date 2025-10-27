# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Any, List

import serializer.tosa_serializer as ts

import torch

from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
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
        validate_num_inputs(self.target, inputs, 3)
        validate_same_dtype(self.target, [inputs[0], output], ts)
        validate_valid_dtype(
            self.target,
            [inputs[0], output],
            [
                ts.DType.INT8,
                ts.DType.INT32,
                ts.DType.FP32,
                ts.DType.BOOL,
            ],
            output.tosa_spec,
        )

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

        self._serialize_operator(
            node,
            tosa_graph,
            ts.TosaOp.Op().PAD,
            [inputs[0].name, padding.name, pad_const.name],
            [output.name],
        )
