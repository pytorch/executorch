# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-unsafe
from typing import cast, Dict

import numpy as np
import serializer.tosa_serializer as ts
import torch
import torch.fx

# pyre-fixme[21]: 'Could not find a module corresponding to import `executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass`.'
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
)
from executorch.backends.arm.operators.node_visitor import NodeVisitor
from executorch.backends.arm.tosa_mapping import map_dtype, TosaArg
from executorch.backends.arm.tosa_quant_utils import (
    get_quantized_node_output_dtype,
    is_node_quantized,
)
from executorch.backends.arm.tosa_specification import TosaSpecification
from executorch.backends.arm.tosa_utils import (
    getNodeArgs,
    is_bias_node_for_quantized_conv,
    tosa_shape,
)
from torch.export.exported_program import ExportedProgram


def process_call_function(
    node: torch.fx.Node,
    tosa_graph: ts.TosaSerializer,
    node_visitors: Dict[str, NodeVisitor],
    tosa_spec: TosaSpecification,
):
    # Unpack arguments and convert
    inputs = getNodeArgs(node)

    # Convert output (this node itself)
    output = TosaArg(node)

    is_quant_node = is_node_quantized(node)
    if is_quant_node:
        output_dtype = map_dtype(get_quantized_node_output_dtype(node))
    else:
        output_dtype = output.dtype
    tosa_graph.currRegion.currBasicBlock.addTensor(
        output.name,
        tosa_shape(output.shape, output.dim_order),
        output_dtype,
    )

    # Visiting each Node
    # pyre-ignore[16]: Undefined attribute.
    if node.target.__name__ in node_visitors:
        # pyre-ignore[16]: Undefined attribute.
        node_visitors[node.target.__name__].define_node(
            node,
            tosa_graph,
            inputs,
            output,
            is_quant_node,
        )
    else:
        raise RuntimeError(f"Unknown operator {node.target} for TOSA : {tosa_spec}")


def process_inputs(
    node: torch.fx.Node,
    tosa_graph: ts.TosaSerializer,
    tosa_spec: TosaSpecification,
):
    """Serialize an input node"""
    # inputs need to be in default dim_order (contiguous memory format)
    meta = node.meta["val"]
    if meta.dim_order() != tuple(range(meta.dim())):
        raise RuntimeError(
            f"Arm backend only supports contiguous memory format for inputs. "
            f"Expected dim_order: {tuple(range(meta.dim()))}, but got: {meta.dim_order()} for node {node.name}"
        )
    inputs = [TosaArg(node)]
    input_shape = inputs[0].shape
    input_dim_order = inputs[0].dim_order
    tensor = ts.TosaSerializerTensor(
        inputs[0].name,
        tosa_shape(input_shape, input_dim_order),
        (
            map_dtype(get_quantized_node_output_dtype(node))
            if is_node_quantized(node)
            else inputs[0].dtype
        ),
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
    """
    consumer_node = list(node.users)[0]
    (
        input_node,
        weight_node,
        _,
    ) = consumer_node.all_input_nodes

    input_qargs = get_input_qparams(  # pyre-ignore[16]: Module `executorch.backends.arm` has no attribute `_passes`.
        consumer_node
    )

    input_node_scale = input_qargs[0].scale
    weight_node_scale = input_qargs[1].scale
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
    tosa_spec: TosaSpecification,
):
    """Serialize bias and non-quantized weights"""
    inputs = [TosaArg(node)]
    parameter_name = edge_program.graph_signature.inputs_to_parameters[node.name]
    parameter_data = edge_program.state_dict[parameter_name]

    assert isinstance(parameter_data, torch.Tensor), "Expect Attr to be tensor"
    parameter_values = parameter_data.detach().numpy()

    if is_bias_node_for_quantized_conv(node):
        # BI bias
        assert tosa_spec.support_integer(), f"{tosa_spec} doesnt't support integer"
        process_quantized_bias(node, tosa_graph, parameter_values)
    else:
        # MI weights or bias
        if inputs[0].dtype == torch.float32:
            assert tosa_spec.support_float(), f"{tosa_spec} doesn't support float"

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
    tosa_spec: TosaSpecification,
):
    """Wrapper for processing and serializing all types of placeholders"""
    assert node.name == node.target, "Expect placeholder name and target to match"
    assert 0 == len(node.args), "Can't handle default input values"

    if node.name in edge_program.graph_signature.user_inputs:
        process_inputs(node, tosa_graph, tosa_spec)
    elif node.name in edge_program.graph_signature.inputs_to_parameters:
        process_inputs_to_parameters(node, tosa_graph, edge_program, tosa_spec)
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


def process_output(
    node: torch.fx.Node,
    tosa_graph: ts.TosaSerializer,
):
    for output in cast(tuple[torch.fx.Node, ...], node.args[0]):
        tosa_graph.addOutputTensor(
            tosa_graph.currRegion.currBasicBlock.tensors[output.name]
        )
