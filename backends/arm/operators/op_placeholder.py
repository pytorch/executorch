# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import serializer.tosa_serializer as ts
import torch
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_quant_utils import (
    dq_q_ops,
    get_quant_node_args,
    is_quant_arg,
    q_op,
)
from executorch.backends.arm.tosa_utils import getNodeArgs, is_bias_node_for_addmm
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export.exported_program import ExportedProgram


def process_placeholder(
    node: torch.fx.Node, tosa_graph: ts.TosaSerializer, edge_program: ExportedProgram
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

        # Check if they're for quantized nodes
        consumer_node = list(node.users)[0]
        if consumer_node.target in dq_q_ops:
            _, weight_node_scale, weight_node_zp, _, _, _ = getNodeArgs(consumer_node)

            int8_max = np.iinfo(np.int8).max
            int8_min = np.iinfo(np.int8).min
            parameter_values_quantized = (
                ((parameter_values / weight_node_scale.number) + weight_node_zp.number)
                .round()
                .clip(int8_min, int8_max)
                .astype(np.int8)
            )
            tosa_graph.addConst(
                inputs[0].shape,
                ts.DType.INT8,
                parameter_values_quantized,
                name=out,
            )
        elif is_bias_node_for_addmm(node):
            (
                _,
                input_node,
                weight_node_permuted,
            ) = consumer_node.all_input_nodes
            weight_node = weight_node_permuted.all_input_nodes[0]

            if input_node.target == exir_ops.edge.aten.view_copy.default:
                input_node_scale, _ = get_quant_node_args(input_node.all_input_nodes[0])
            else:
                input_node_scale, _ = get_quant_node_args(input_node)

            weight_node_scale, weight_node_zp = get_quant_node_args(weight_node)

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
            (
                input_node,
                weight_node,
                bias_node,
            ) = consumer_node.all_input_nodes

            input_node_scale, _ = get_quant_node_args(input_node)
            weight_node_scale, _ = get_quant_node_args(weight_node)

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
            tosa_graph.addConst(
                inputs[0].shape, inputs[0].dtype, parameter_values, name=out
            )

    elif out in edge_program.graph_signature.inputs_to_buffers:
        parameter_name = edge_program.graph_signature.inputs_to_buffers[node.name]
        p_data = edge_program.state_dict[parameter_name]

        assert isinstance(p_data, torch.Tensor), "Expect Attr to be tensor"
        buffer_values = p_data.detach().numpy()
        tosa_graph.addConst(inputs[0].shape, inputs[0].dtype, buffer_values, name=out)
    else:
        tensor = ts.TosaSerializerTensor(
            inputs[0].name,
            inputs[0].shape,
            ts.DType.INT8 if is_quant_arg(node) else inputs[0].dtype,
            data=None,
            placeholderFilename=inputs[0].name + ".npy",
        )
        tosa_graph.addInputTensor(tensor)
