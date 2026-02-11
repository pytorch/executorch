# Copyright 2023-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tosa_serializer as ts

"""Provide a visitor for lowering 2D convolution to TOSA (INT/FP)."""

from typing import Any, List

import torch
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_valid_dtype,
)
from executorch.backends.arm.operators.ops_quant_utils import add_input_weight_zp_consts
from executorch.backends.arm.tosa.mapping import TosaArg


@register_node_visitor
class Conv2dVisitor(NodeVisitor):
    """Provide a visitor that serializes TOSA ``CONV2D``."""

    target = "tosa.CONV2D.default"

    def __init__(self, *args):
        super().__init__(*args)

    def _get_tosa_op(self):
        return ts.Op.CONV2D

    def _get_attr_func(self, attr):
        return attr.Conv2dAttribute

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        """Define the TOSA CONV2D/DEPTHWISE_CONV2D operator."""

        input, weight, bias, stride, pad, dilation = inputs
        validate_num_inputs(self.target, inputs, 6)

        valid_input_dtypes = []
        if self.tosa_spec.support_float():
            valid_input_dtypes.extend([ts.DType.FP16, ts.DType.FP32])
        if self.tosa_spec.support_integer():
            valid_input_dtypes.append(ts.DType.INT8)

        if self.tosa_spec.support_extension("int16"):
            valid_input_dtypes.append(ts.DType.INT16)
            # Check constraints for int16 activations
            if inputs[0].dtype == ts.DType.INT16:
                validate_valid_dtype(
                    self.target, [inputs[1]], [ts.DType.INT8], self.tosa_spec
                )
                validate_valid_dtype(
                    self.target, [inputs[2]], [ts.DType.INT48], self.tosa_spec
                )
        if self.tosa_spec.support_extension("bf16"):
            valid_input_dtypes.append(ts.DType.BF16)

        validate_valid_dtype(
            self.target,
            [inputs[0]],
            valid_input_dtypes,
            self.tosa_spec,
        )

        # Get the attributes of convolution.
        pad_attr = pad.special
        stride_attr = stride.special
        dilation_attr = dilation.special

        conv2d_output_name = output.name
        acc_type = output.dtype
        if output.dtype in [ts.DType.BF16, ts.DType.FP16]:
            # Accumulate BF16, FP16 inputs in FP32 for better precision.
            acc_type = ts.DType.FP32

        input_zp_name, weight_zp_name = add_input_weight_zp_consts(
            tosa_graph, node, inputs, conv2d_output_name
        )

        tosa_op = self._get_tosa_op()

        attr = ts.TosaSerializerAttribute()
        self._get_attr_func(attr)(
            pad=pad_attr,
            stride=stride_attr,
            dilation=dilation_attr,
            local_bound=False,
            acc_type=acc_type,
        )

        self._serialize_operator(
            node,
            tosa_graph,
            tosa_op,
            [
                input.name,
                weight.name,
                bias.name,
                input_zp_name,
                weight_zp_name,
            ],
            [conv2d_output_name],
            attr,
        )
