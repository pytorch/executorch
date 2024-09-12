# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

import serializer.tosa_serializer as ts
import torch
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_quant_utils import build_rescale, get_quant_node_args

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

        input_zp = 0
        if is_quant_node:
            input_node = node.all_input_nodes[1]
            # rank > 2 linear layer
            if input_node.target == exir_ops.edge.aten.view_copy.default:
                quant_node = input_node.all_input_nodes[0]
            else:
                quant_node = input_node
            input_zp = get_quant_node_args(quant_node).zp
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
                input_scale = get_quant_node_args(quant_node).scale
                consumer_node = list(node.users)[0]
                consumer_consumer_node = list(consumer_node.users)[0]
                quant_args = get_quant_node_args(consumer_consumer_node)
                consumer_node_scale = quant_args.scale
                consumer_node_node_zp = quant_args.zp
            else:
                input_scale = get_quant_node_args(input_node).scale
                consumer_node = list(node.users)[0]
                quant_args = get_quant_node_args(consumer_node)
                consumer_node_scale = quant_args.scale
                consumer_node_node_zp = quant_args.zp

            weight_node_q_node = weight_node.all_input_nodes[0]
            weight_scale = get_quant_node_args(weight_node_q_node).scale

            output_rescale_scale = (input_scale * weight_scale) / consumer_node_scale

            reshaped_res = tosa_graph.addIntermediate(result_shape, ts.DType.INT32)
            build_reshape(tosa_graph, conv2d_res.name, result_shape, reshaped_res.name)

            build_rescale(
                tosa_fb=tosa_graph,
                scale=output_rescale_scale,
                input_node=reshaped_res,
                output_name=output.name,
                output_type=ts.DType.INT8,
                output_shape=reshaped_res.shape,
                input_zp=0,
                output_zp=consumer_node_node_zp,
                is_double_round=False,
            )

        else:
            # non-quantized case
            build_reshape(tosa_graph, conv2d_res.name, result_shape, output.name)
