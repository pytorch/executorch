# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import numpy as np
import serializer.tosa_serializer as ts
import torch.fx
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_quant_utils import (
    get_quant_arg_dtype,
    get_quant_node_args,
    is_quant_arg,
)
from executorch.backends.arm.tosa_utils import (
    is_bias_node_for_quantized_addmm,
    is_bias_node_for_quantized_conv,
    tosa_shape,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export.exported_program import ExportedProgram


def process_inputs(
    node: torch.fx.Node,
    tosa_graph: ts.TosaSerializer,
):
    """Serialize an input node"""
    inputs = [TosaArg(node)]
    input_shape = inputs[0].shape
    input_dim_order = inputs[0].dim_order
    tensor = ts.TosaSerializerTensor(
        inputs[0].name,
        tosa_shape(input_shape, input_dim_order),
        get_quant_arg_dtype(node) if is_quant_arg(node) else inputs[0].dtype,
        data=None,
        placeholderFilename=inputs[0].name + ".npy",
    )
    tosa_graph.addInputTensor(tensor)


def process_quantized_bias(
    node: torch.fx.Node,
    tosa_graph: ts.TosaSerializer,
    parameter_values,
):
    """
    Serialize bias node that needs to be quantized.
    This can be either an addmm or conv bias node.
    """
    consumer_node = list(node.users)[0]
    if is_bias_node_for_quantized_addmm(node):
        (
            _,
            input_node,
            weight_node_permuted,
        ) = consumer_node.all_input_nodes

        weight_node = weight_node_permuted.all_input_nodes[0]
        if input_node.target == exir_ops.edge.aten.view_copy.default:
            input_node = input_node.all_input_nodes[0]
    else:
        (
            input_node,
            weight_node,
            _,
        ) = consumer_node.all_input_nodes

    input_node_scale = get_quant_node_args(input_node).scale
    weight_node_scale = get_quant_node_args(weight_node).scale
    bias_values_quantized = (
        (parameter_values / (input_node_scale * weight_node_scale))
        .round()
        .astype(np.int32)
    )

    tosa_graph.addConst(
        bias_values_quantized.shape,
        ts.DType.INT32,
        bias_values_quantized,
        name=node.name,
    )


def process_inputs_to_parameters(
    node: torch.fx.Node,
    tosa_graph: ts.TosaSerializer,
    edge_program: ExportedProgram,
):
    """Serialize bias and non-quantized weights"""
    inputs = [TosaArg(node)]
    parameter_name = edge_program.graph_signature.inputs_to_parameters[node.name]
    parameter_data = edge_program.state_dict[parameter_name]

    assert isinstance(parameter_data, torch.Tensor), "Expect Attr to be tensor"
    parameter_values = parameter_data.detach().numpy()

    if is_bias_node_for_quantized_addmm(node) or is_bias_node_for_quantized_conv(node):
        # BI bias
        process_quantized_bias(node, tosa_graph, parameter_values)
    else:
        # MI weights or bias
        parameter_values = np.transpose(parameter_values, inputs[0].dim_order)

        tosa_graph.addConst(
            parameter_values.shape, inputs[0].dtype, parameter_values, name=node.name
        )


def process_inputs_to_buffers(
    node: torch.fx.Node,
    tosa_graph: ts.TosaSerializer,
    edge_program: ExportedProgram,
):
    """Serialize quantized weights"""
    inputs = [TosaArg(node)]
    buffer_name = edge_program.graph_signature.inputs_to_buffers[node.name]
    buffer_data = edge_program.state_dict[buffer_name]

    assert isinstance(buffer_data, torch.Tensor), "Expect Attr to be tensor"
    buffer_values = buffer_data.detach().numpy()

    # TODO: fragile code for temporary fix
    # the mean and var tensors are also stored here but they have shape (1, )
    # we only transpose weights here
    buffer_values = np.transpose(buffer_values, inputs[0].dim_order)

    tosa_graph.addConst(
        buffer_values.shape, inputs[0].dtype, buffer_values, name=node.name
    )


def process_inputs_to_lifted_tensor_constants(
    node: torch.fx.Node,
    tosa_graph: ts.TosaSerializer,
    edge_program: ExportedProgram,
):
    arg = TosaArg(node)
    tensor_name = edge_program.graph_signature.inputs_to_lifted_tensor_constants[
        arg.name
    ]
    tensor = edge_program.tensor_constants[tensor_name]
    tensor_data = tensor.detach().numpy()

    tosa_graph.addConst(tensor_data.shape, arg.dtype, tensor_data, name=arg.name)


def process_placeholder(
    node: torch.fx.Node,
    tosa_graph: ts.TosaSerializer,
    edge_program: ExportedProgram,
):
    """Wrapper for processing and serializing all types of placeholders"""
    assert node.name == node.target, "Expect placeholder name and target to match"
    assert 0 == len(node.args), "Can't handle default input values"

    if node.name in edge_program.graph_signature.user_inputs:
        process_inputs(node, tosa_graph)
    elif node.name in edge_program.graph_signature.inputs_to_parameters:
        process_inputs_to_parameters(node, tosa_graph, edge_program)
    elif node.name in edge_program.graph_signature.inputs_to_buffers:
        process_inputs_to_buffers(node, tosa_graph, edge_program)
    elif node.name in edge_program.graph_signature.inputs_to_lifted_tensor_constants:
        process_inputs_to_lifted_tensor_constants(node, tosa_graph, edge_program)
    elif node.name in edge_program.graph_signature.inputs_to_lifted_custom_objs:
        raise NotImplementedError(
            "Placeholder is of type 'lifted custom object' which is not supported."
        )
    else:
        raise RuntimeError(f"Placeholder '{node.name}' is of unknown type.")
