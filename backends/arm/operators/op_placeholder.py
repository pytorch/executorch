# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import serializer.tosa_serializer as ts
import torch
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_quant_utils import (
    get_quant_node_args,
    is_quant_arg,
    q_op,
)
from executorch.backends.arm.tosa_utils import (
    is_bias_node_for_addmm,
    is_consumer_node_depthwise_conv2d,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export.exported_program import ExportedProgram


def process_placeholder(
    node: torch.fx.Node,
    tosa_graph: ts.TosaSerializer,
    edge_program: ExportedProgram,
    permute_memory_to_nhwc: bool,
):
    assert node.name == node.target, "Expect placeholder name and target to match"
    assert 0 == len(node.args), "Can't handle default input values"
    inputs = [TosaArg(node)]
    out = node.name

    if out in edge_program.graph_signature.inputs_to_parameters:
        parameter_name = edge_program.graph_signature.inputs_to_parameters[node.name]
        p_data = edge_program.state_dict[parameter_name]

        assert isinstance(p_data, torch.Tensor), "Expect Attr to be tensor"
        parameter_values = p_data.detach().numpy()
        consumer_node = list(node.users)[0]

        if is_bias_node_for_addmm(node):
            # Cases for:
            # - BI_AddMM_bias
            (
                _,
                input_node,
                weight_node_permuted,
            ) = consumer_node.all_input_nodes
            weight_node = weight_node_permuted.all_input_nodes[0]

            if input_node.target == exir_ops.edge.aten.view_copy.default:
                input_node_scale = get_quant_node_args(
                    input_node.all_input_nodes[0]
                ).scale
            else:
                input_node_scale = get_quant_node_args(input_node).scale

            weight_node_scale = get_quant_node_args(weight_node).scale

            bias_values_quantized = (
                (parameter_values / (input_node_scale * weight_node_scale))
                .round()
                .astype(np.int32)
            )

            tosa_graph.addConst(
                inputs[0].shape,
                ts.DType.INT32,
                bias_values_quantized,
                name=out,
            )
        elif (
            consumer_node.target == exir_ops.edge.aten.convolution.default
            and list(consumer_node.users)[0].target == q_op
        ):
            # Cases for:
            # - BI_Conv2d_bias
            # - BI_DepthwiseConv2d_bias
            (
                input_node,
                weight_node,
                bias_node,
            ) = consumer_node.all_input_nodes

            input_node_scale = get_quant_node_args(input_node).scale
            weight_node_scale = get_quant_node_args(weight_node).scale

            bias_scales = input_node_scale * weight_node_scale
            parameter_values_quantized = (
                (parameter_values / bias_scales).round().astype(np.int32)
            )

            tosa_graph.addConst(
                inputs[0].shape,
                ts.DType.INT32,
                parameter_values_quantized,
                name=out,
            )
        else:
            # Cases for:
            # - MI_AddMM_bias
            # - MI_AddMM_weight
            # - MI_Conv2d_non_bias_weight
            # - MI_Conv2d_weight
            # - MI_Conv2d_bias
            # - MI_DepthwiseConv2d_weight
            # - MI_DepthwiseConv2d_bias
            if (
                permute_memory_to_nhwc
                and len(parameter_values.shape) == 4
                and is_consumer_node_depthwise_conv2d(node)
            ):
                # For more details on TOSA depthwise_conv2d:
                # https://www.mlplatform.org/tosa/tosa_spec.html#_depthwise_conv2d
                HWCM_Order = [2, 3, 0, 1]
                parameter_values = np.transpose(parameter_values, HWCM_Order)
            elif permute_memory_to_nhwc and len(parameter_values.shape) == 4:
                # For regular conv2d case
                NHWC_Order = (0, 2, 3, 1)
                parameter_values = np.transpose(parameter_values, NHWC_Order)

            tosa_graph.addConst(
                parameter_values.shape, inputs[0].dtype, parameter_values, name=out
            )

    elif out in edge_program.graph_signature.inputs_to_buffers:
        # Cases for:
        # - BI_AddMM_weight
        # - BI_Conv2d_non_bias_weight
        # - BI_Conv2d_weight
        # - BI_DepthwiseConv2d_weight
        # - MI_BatchNorm_variance
        # - MI_BatchNorm_mean
        parameter_name = edge_program.graph_signature.inputs_to_buffers[node.name]
        p_data = edge_program.state_dict[parameter_name]

        assert isinstance(p_data, torch.Tensor), "Expect Attr to be tensor"
        buffer_values = p_data.detach().numpy()

        # TODO: fragile code for temporary fix
        # the mean and var tensors are also stored here but they have shape (1, )
        # we only transpose weights here
        if permute_memory_to_nhwc and is_consumer_node_depthwise_conv2d(
            list(node.users)[0]
        ):
            # For more details on TOSA depthwise_conv2d:
            # https://www.mlplatform.org/tosa/tosa_spec.html#_depthwise_conv2d
            HWCM_Order = (2, 3, 0, 1)
            buffer_values = np.transpose(buffer_values, HWCM_Order)
        elif permute_memory_to_nhwc and len(buffer_values.shape) == 4:
            # For regular conv2d case
            NHWC_Order = (0, 2, 3, 1)
            buffer_values = np.transpose(buffer_values, NHWC_Order)

        tosa_graph.addConst(
            buffer_values.shape, inputs[0].dtype, buffer_values, name=out
        )
    else:
        # Cases for all the input tensors of rank4
        if permute_memory_to_nhwc and len(inputs[0].shape) == 4:
            NHWC_Order = [0, 2, 3, 1]
            input_shape = [inputs[0].shape[i] for i in NHWC_Order]
        else:
            input_shape = inputs[0].shape
        tensor = ts.TosaSerializerTensor(
            inputs[0].name,
            input_shape,
            ts.DType.INT8 if is_quant_arg(node) else inputs[0].dtype,
            data=None,
            placeholderFilename=inputs[0].name + ".npy",
        )
        tosa_graph.addInputTensor(tensor)
