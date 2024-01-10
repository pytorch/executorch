# Copyright 2023-2024 Arm Limited and/or its affiliates.
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
from executorch.backends.arm.tosa_quant_utils import (
    compute_multiplier_and_shift,
    get_quant_node_args,
)

from executorch.backends.arm.tosa_utils import build_reshape
from executorch.exir.dialects._ops import ops as exir_ops
from serializer.tosa_serializer import TosaOp


@register_node_visitor
class AddmmVisitor(NodeVisitor):
    target = "aten.addmm.default"

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
        bias, input, weight = inputs

        N = input.shape[0]
        input_channels = input.shape[1]
        output_channels = weight.shape[1]

        input_new_shape = (N, 1, 1, input_channels)
        input_reshaped = tosa_graph.addIntermediate(
            input_new_shape,
            ts.DType.INT8 if is_quant_node else input.dtype,
        )

        build_reshape(tosa_graph, input.name, input_new_shape, input_reshaped.name)

        weight_new_shape = (output_channels, 1, 1, input_channels)
        weight_reshaped = tosa_graph.addIntermediate(
            weight_new_shape,
            ts.DType.INT8 if is_quant_node else weight.dtype,
        )

        build_reshape(tosa_graph, weight.name, weight_new_shape, weight_reshaped.name)

        # Get the attributes of convolution.
        attr = ts.TosaSerializerAttribute()
        pad_attr = [0, 0, 0, 0]
        stride_attr = [1, 1]
        dilation_attr = [1, 1]

        input_zp = -128 if is_quant_node else 0
        attr.ConvAttribute(
            pad=pad_attr,
            stride=stride_attr,
            dilation=dilation_attr,
            input_zp=input_zp,
            weight_zp=0,
            local_bound=False,
        )

        conv2d_output_shape = (N, 1, 1, output_channels)
        conv2d_res = tosa_graph.addIntermediate(
            conv2d_output_shape,
            ts.DType.INT32 if is_quant_node else output.dtype,
        )

        # U55 doesn't support tosa.matmul and tosa.fully_connected will be deprecated
        # TOSA Conv2d input is NHWC and weights are in OHWI
        tosa_graph.addOperator(
            TosaOp.Op().CONV2D,
            [
                input_reshaped.name,
                weight_reshaped.name,
                bias.name,
            ],
            [conv2d_res.name],
            attr,
        )

        result_shape = (N, output_channels)

        if is_quant_node:
            # Read inputs' parent nodes
            _, input_node, weight_node = node.all_input_nodes

            # rank > 2 linear layer
            if input_node.target == exir_ops.edge.aten.view_copy.default:
                quant_node = input_node.all_input_nodes[0]
                input_scale, _ = get_quant_node_args(quant_node)
                consumer_node = list(node.users)[0]
                consumer_consumer_node = list(consumer_node.users)[0]
                (
                    consumer_node_scale,
                    consumer_node_node_zp,
                ) = get_quant_node_args(consumer_consumer_node)

            else:
                input_scale, _ = get_quant_node_args(input_node)
                consumer_node = list(node.users)[0]
                (
                    consumer_node_scale,
                    consumer_node_node_zp,
                ) = get_quant_node_args(consumer_node)

            weight_node_q_node = weight_node.all_input_nodes[0]
            weight_scale, _ = get_quant_node_args(weight_node_q_node)

            output_rescale_scale = (input_scale * weight_scale) / consumer_node_scale
            (
                multiplier_output,
                shift_output,
            ) = compute_multiplier_and_shift(output_rescale_scale)

            attr_rescale_output = ts.TosaSerializerAttribute()
            attr_rescale_output.RescaleAttribute(
                input_zp=0,
                output_zp=consumer_node_node_zp,
                multiplier=[multiplier_output],
                shift=[shift_output],
                scale32=True,
                double_round=True,
                per_channel=False,
                input_unsigned=False,
                output_unsigned=False,
            )

            reshaped_res = tosa_graph.addIntermediate(result_shape, ts.DType.INT32)
            build_reshape(tosa_graph, conv2d_res.name, result_shape, reshaped_res.name)

            tosa_graph.addOperator(
                TosaOp.Op().RESCALE,
                [reshaped_res.name],
                [output.name],
                attr_rescale_output,
            )

        else:
            # non-quantized case
            build_reshape(tosa_graph, conv2d_res.name, result_shape, output.name)
