# Copyright 2023-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tosa_serializer as ts

"""Provide a visitor for lowering 2D convolution to TOSA (INT/FP)."""

from typing import Any, List

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
    validate_valid_dtype,
)
from executorch.backends.arm.tosa.mapping import TosaArg
from executorch.backends.arm.tosa.specification import TosaSpecification


@register_node_visitor
class Conv2dVisitor(NodeVisitor):
    """Provide a visitor that serializes TOSA ``CONV2D``."""

    target = "tosa.CONV2D.default"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
    ]

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
            valid_input_dtypes.append(ts.DType.FP32)
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

        input_zp = 0
        if inputs[0].dtype in (ts.DType.INT8, ts.DType.INT16):
            # int8 and int16 input requires quantization information
            input_qparams = get_input_qparams(node)
            input_zp = input_qparams[0].get_zp_per_tensor()

        weight_zp = 0
        if inputs[1].dtype == ts.DType.INT8:
            # int8 weights requires quantization information
            input_qparams = get_input_qparams(node)
            weight_zp = input_qparams[1].zp  # type: ignore[assignment]

        conv2d_output_name = output.name
        acc_type = output.dtype
        if output.dtype == ts.DType.BF16:
            # Accumulate BF16 inputs in FP32 for better precision per TOSA BF16 extension.
            acc_type = ts.DType.FP32

        tosa_graph.addConst(
            [1], inputs[0].dtype, [input_zp], name=f"{conv2d_output_name}_input_zp"
        )
        tosa_graph.addConst(
            [1],
            inputs[1].dtype,
            weight_zp,
            name=f"{conv2d_output_name}_weight_zp",
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
                f"{conv2d_output_name}_input_zp",
                f"{conv2d_output_name}_weight_zp",
            ],
            [conv2d_output_name],
            attr,
        )
