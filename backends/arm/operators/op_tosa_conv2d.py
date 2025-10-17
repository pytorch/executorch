# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
"""Provide a visitor for lowering 2D convolution to TOSA (INT/FP)."""

import itertools
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
    validate_valid_dtype,
)
from executorch.backends.arm.tosa.mapping import TosaArg
from executorch.backends.arm.tosa.quant_utils import build_rescale
from executorch.backends.arm.tosa.specification import Tosa_1_00, TosaSpecification
from executorch.backends.arm.tosa.utils import tosa_shape


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
        import serializer.tosa_serializer as ts  # type: ignore

        return ts.TosaOp.Op().CONV2D

    def _get_attr_func(self, attr):
        return attr.Conv2dAttribute

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        """Define the TOSA CONV2D/DEPTHWISE_CONV2D operator and post-rescale."""
        import serializer.tosa_serializer as ts  # type: ignore
        from tosa.RoundingMode import RoundingMode  # type: ignore

        input, weight, bias, stride, pad, dilation, _, _, group = inputs
        validate_num_inputs(self.target, inputs, 9)

        valid_input_dtypes = []
        if self.tosa_spec.support_float():
            valid_input_dtypes.append(ts.DType.FP32)
        if self.tosa_spec.support_integer():
            valid_input_dtypes.append(ts.DType.INT8)

        if isinstance(self.tosa_spec, Tosa_1_00) and self.tosa_spec.support_extension(
            "int16"
        ):
            valid_input_dtypes.append(ts.DType.INT16)
            # Check constraints for int16 activations
            if inputs[0].dtype == ts.DType.INT16:
                validate_valid_dtype(
                    self.target, [inputs[1]], [ts.DType.INT8], self.tosa_spec
                )
                validate_valid_dtype(
                    self.target, [inputs[2]], [ts.DType.INT48], self.tosa_spec
                )

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

        # The output type is int32 when input type is int8.
        if inputs[0].dtype == ts.DType.INT8:
            conv2d_res = tosa_graph.addIntermediate(
                tosa_shape(output.shape, output.dim_order), ts.DType.INT32
            )
            conv2d_output_name = conv2d_res.name
            acc_type = ts.DType.INT32
        elif inputs[0].dtype == ts.DType.INT16:
            conv2d_res = tosa_graph.addIntermediate(
                tosa_shape(output.shape, output.dim_order), ts.DType.INT48
            )
            conv2d_output_name = conv2d_res.name
            acc_type = ts.DType.INT48
        else:
            conv2d_output_name = output.name
            conv2d_res = output
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

        # For quantized convolution, rescale the output value back to the same
        # integer value domain of the next op. Otherwise return float32 output.
        if output.dtype == ts.DType.INT8 or output.dtype == ts.DType.INT16:
            # Get scale_factor from input, weight, and output.
            input_scale = input_qparams[0].get_scale_per_tensor()  # type: ignore[possibly-undefined]  # pyre-ignore [61]
            per_channel_quant = input_qparams[1].per_channel  # pyre-ignore [61]
            if per_channel_quant:
                weight_scale = input_qparams[1].get_scale_per_channel()
            else:
                weight_scale = [
                    input_qparams[1].get_scale_per_tensor()
                ]  # pyre-ignore [61]
            output_qargs = get_output_qparams(node)
            post_conv2d_scale = [
                (inp * w) / out
                for inp, w, out in zip(
                    itertools.cycle([input_scale]),
                    weight_scale,
                    itertools.cycle([output_qargs[0].get_scale_per_tensor()]),
                )
            ]
            build_rescale(
                tosa_fb=tosa_graph,
                scale=post_conv2d_scale,
                input_node=conv2d_res,  # type: ignore[possibly-undefined]
                output_name=output.name,
                output_type=output.dtype,
                input_zp=[0],
                output_zp=[output_qargs[0].get_zp_per_tensor()],
                per_channel=per_channel_quant,
                rounding_mode=RoundingMode.SINGLE_ROUND,
            )
