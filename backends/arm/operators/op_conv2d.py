# Copyright 2023 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

import serializer.tosa_serializer as ts
import torch
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_quant_utils import buildRescaleOpConvOutput
from executorch.backends.arm.tosa_utils import (
    buildReshape,
    getNodeArgs,
    transpose_helper,
)

from serializer.tosa_serializer import TosaOp


@register_node_visitor
class Conv2dVisitor(NodeVisitor):
    target = "aten.convolution.default"

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
        is_quant_node: bool,
    ) -> None:
        input, weight, bias, stride, pad, dilation, _, _, group = inputs

        # Currently only int8 is supported in quantized types.
        actual_out_type = ts.DType.INT8 if is_quant_node else output.dtype

        ## Transpose input tensor to NHWC_Order for TOSA
        NHWC_Order = [0, 2, 3, 1]
        input_transposed = transpose_helper(
            tosa_graph, input, NHWC_Order, actual_out_type
        )

        # Get the attributes of convolution.
        attr = ts.TosaSerializerAttribute()
        pad_attr = [val for val in pad.special for _ in (0, 1)]
        stride_attr = stride.special
        dilation_attr = dilation.special
        attr.ConvAttribute(pad_attr, stride_attr, dilation_attr, 0, 0)

        # Non-bias case.
        if len(node.all_input_nodes) == 2:
            # Create a zero bias tensor if not presented
            out_channels = weight.shape[0]
            bias_name = "bias" + node.name.split("default", 1)[1]
            bias = tosa_graph.addConst(
                [out_channels],
                ts.DType.INT32 if is_quant_node else output.dtype,
                [0] * out_channels,
                name=bias_name,
            )

        if group.number > 1:
            """Depthwise convolution case"""
            # Given input.shape is (N, Ci, H, W), and weight.shape is (Co, Ci/G, H, W)
            in_channels = input.shape[1]
            out_channels = weight.shape[0]

            # Reshape torch shape format of weight tensor to tosa required format.
            # https://www.mlplatform.org/tosa/tosa_spec.html#_depthwise_conv2d
            m_length = int(round(out_channels / in_channels))
            weight_post_shape = (
                in_channels,
                m_length,
                weight.shape[2],
                weight.shape[3],
            )

            weight_reshaped = tosa_graph.addIntermediate(
                weight_post_shape,
                ts.DType.INT8 if is_quant_node else weight.dtype,
            )

            buildReshape(
                tosa_graph, weight.name, weight_post_shape, weight_reshaped.name
            )

            # Transpose weight to [KH, KW, C, M]
            weight_HWCM_Order = [2, 3, 0, 1]
            weight_transposed = transpose_helper(
                tosa_graph,
                weight_reshaped,
                weight_HWCM_Order,
                ts.DType.INT8 if is_quant_node else weight.dtype,
            )

            ## TOSA output shape is [N, H, W, C*M]
            NHWO_Order = [0, 2, 3, 1]
            out_shape_TOSA_Depthwise_CONV2D = [output.shape[i] for i in NHWO_Order]

            conv2d_res = tosa_graph.addIntermediate(
                out_shape_TOSA_Depthwise_CONV2D,
                ts.DType.INT32 if is_quant_node else output.dtype,
            )
            tosa_graph.addOperator(
                TosaOp.Op().DEPTHWISE_CONV2D,
                [
                    input_transposed.name,
                    weight_transposed.name,
                    bias.name,
                ],
                [conv2d_res.name],
                attr,
            )
        else:
            """Regular convolution case"""
            # Transpose weight to [OC, H, W, IC]
            weight_CHWC_Order = [0, 2, 3, 1]
            weight_transposed = transpose_helper(
                tosa_graph,
                weight,
                weight_CHWC_Order,
                actual_out_type,
            )

            ## TOSA output shape is [NHWO]
            NHWO_Order = [0, 2, 3, 1]
            out_shape_TOSA_CONV2D = [output.shape[i] for i in NHWO_Order]

            # The output type is int32 when input type is int8.
            conv2d_res = tosa_graph.addIntermediate(
                out_shape_TOSA_CONV2D,
                ts.DType.INT32 if is_quant_node else output.dtype,
            )
            tosa_graph.addOperator(
                TosaOp.Op().CONV2D,
                [
                    input_transposed.name,
                    weight_transposed.name,
                    bias.name,
                ],
                [conv2d_res.name],
                attr,
            )

        ## Torch output shape is [NOHW]
        NOHW_Order = [0, 3, 1, 2]
        attr_output_transpose = ts.TosaSerializerAttribute()
        attr_output_transpose.TransposeAttribute(NOHW_Order)

        # For quantized convolution, rescale the output value back to the same
        # integer value domain of the next op. Otherwise return float32 output.
        if is_quant_node:
            # Get scale_factor from input, weight, and output.
            _, input_scale, _, _, _, _ = getNodeArgs(node.args[0])
            _, weight_scale, _, _, _, _ = getNodeArgs(node.args[1])
            _, output_scale, _, _, _, _ = getNodeArgs(list(node.users)[0])

            conv2d_res = buildRescaleOpConvOutput(
                tosa_graph,
                conv2d_res,
                actual_out_type,
                input_scale,
                weight_scale,
                output_scale,
            )

        tosa_graph.addOperator(
            TosaOp.Op().TRANSPOSE,
            [conv2d_res.name],
            [output.name],
            attr_output_transpose,
        )
