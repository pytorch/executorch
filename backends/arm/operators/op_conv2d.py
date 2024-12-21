# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import List

import serializer.tosa_serializer as ts
import torch

# pyre-fixme[21]: 'Could not find a module corresponding to import `executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass`.'
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
    get_output_qparams,
)
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_quant_utils import build_rescale_conv_output
from executorch.backends.arm.tosa_utils import build_reshape, tosa_shape

from serializer.tosa_serializer import TosaOp


@register_node_visitor
class Conv2dVisitor(NodeVisitor):
    target = "aten.convolution.default"

    def __init__(self, *args):
        super().__init__(*args)

    # torch.nn.Conv2d does not require the result of
    # `(input + 2 * pad - dilation * (weight - 1) - 1) / stride`
    # must be an integer, but tosa currently strictly require this property.
    # This function adjusts the pad value to meet the requirement.
    def adjust_pad_if_needed(self, input, weight, stride, pad, dilation):
        mod_remainder = (input + 2 * pad - dilation * (weight - 1) - 1) % stride

        # No need to adjust
        if mod_remainder == 0:
            return pad

        if mod_remainder > pad:
            raise RuntimeError(
                "This case should be handled by the SizeAdjustConv2d pass, is it enabled?"
            )
        return pad - mod_remainder

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
        is_quant_node: bool,
    ) -> None:
        input, weight, bias, stride, pad, dilation, _, _, group = inputs

        # Get the attributes of convolution.
        attr = ts.TosaSerializerAttribute()
        pad_attr = [val for val in pad.special for _ in (0, 1)]
        stride_attr = stride.special
        dilation_attr = dilation.special

        # Adjust the pad value if needed to meet the strict convolution output shape calculation.
        pad_attr[1] = self.adjust_pad_if_needed(
            input.shape[2],
            weight.shape[2],
            stride_attr[0],
            pad_attr[1],
            dilation_attr[0],
        )
        pad_attr[3] = self.adjust_pad_if_needed(
            input.shape[3],
            weight.shape[3],
            stride_attr[1],
            pad_attr[3],
            dilation_attr[1],
        )

        input_zp = 0
        if inputs[0].dtype == ts.DType.INT8:
            # int8 input requires quantization information
            input_qparams = get_input_qparams(node)  # pyre-ignore[16]
            input_zp = input_qparams[0].zp

        attr.ConvAttribute(
            pad=pad_attr,
            stride=stride_attr,
            dilation=dilation_attr,
            input_zp=input_zp,
            weight_zp=0,
            local_bound=False,
        )

        # Non-bias case.
        if len(node.all_input_nodes) == 2:
            # Create a zero bias tensor if not presented
            out_channels = weight.shape[0]
            bias_name = "bias" + node.name.split("default", 1)[1]
            bias_type = output.dtype
            if output.dtype == ts.DType.INT8:
                # Conv is quantized to int8, but the TOSA operator has
                # output type int32, and the bias must be the same type
                # as the TOSA output type
                bias_type = ts.DType.INT32
            bias = tosa_graph.addConst(
                [out_channels],
                bias_type,
                [0] * out_channels,
                name=bias_name,
            )

        # The output type is int32 when input type is int8.
        conv2d_output_name = output.name
        if output.dtype == ts.DType.INT8:
            conv2d_res = tosa_graph.addIntermediate(
                tosa_shape(output.shape, output.dim_order), ts.DType.INT32
            )
            conv2d_output_name = conv2d_res.name

        # Given input.shape is (N, Ci, H, W), and weight.shape is (Co, Ci/G, H, W)
        in_channels = input.shape[1]
        out_channels = weight.shape[0]
        if (in_channels == group.number) and (out_channels % in_channels) == 0:
            """Depthwise convolution case"""
            # Reshape torch shape format of weight tensor to tosa required format.
            # https://www.mlplatform.org/tosa/tosa_spec.html#_depthwise_conv2d
            m_length = int(out_channels / in_channels)
            weight_post_shape = (
                weight.shape[2],
                weight.shape[3],
                in_channels,
                m_length,
            )

            weight_reshaped = tosa_graph.addIntermediate(
                weight_post_shape,
                weight.dtype,
            )
            build_reshape(
                tosa_graph, weight.name, weight_post_shape, weight_reshaped.name
            )
            tosa_op = TosaOp.Op().DEPTHWISE_CONV2D
            weight_name = weight_reshaped.name
        else:
            """Regular convolution case"""
            tosa_op = TosaOp.Op().CONV2D
            weight_name = weight.name

        tosa_graph.addOperator(
            tosa_op,
            [
                input.name,
                weight_name,
                bias.name,
            ],
            [conv2d_output_name],
            attr,
        )

        # For quantized convolution, rescale the output value back to the same
        # integer value domain of the next op. Otherwise return float32 output.
        if inputs[0].dtype == ts.DType.INT8:
            # Get scale_factor from input, weight, and output.
            input_scale = input_qparams[0].scale  # pyre-ignore [61]
            weight_scale = input_qparams[1].scale  # pyre-ignore [61]
            output_qargs = get_output_qparams(node)  # pyre-ignore [16]
            build_rescale_conv_output(
                tosa_graph,
                # pyre-fixme[61]: Uninitialized local [61]: Local variable `conv2d_res` is undefined, or not always defined.
                conv2d_res,
                output.name,
                output.dtype,
                input_scale,
                weight_scale,
                output_qargs[0].scale,
                output_qargs[0].zp,
            )
