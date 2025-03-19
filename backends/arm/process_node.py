# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-unsafe
from typing import cast, Dict

import numpy as np
import serializer.tosa_serializer as ts  # type: ignore
import torch
import torch.fx
from executorch.backends.arm.operators.node_visitor import NodeVisitor
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_specification import TosaSpecification
from executorch.backends.arm.tosa_utils import (
    get_node_debug_info,
    getNodeArgs,
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
    try:
        output = TosaArg(node)
    except ValueError as e:
        raise ValueError(
            f"Failed processing call_function:\n{get_node_debug_info(node)}"
            "Is the original torch function supported?"
        ) from e
    tosa_graph.currRegion.currBasicBlock.addTensor(
        output.name, tosa_shape(output.shape, output.dim_order), output.dtype
    )

    # Visiting each Node
    # pyre-ignore[16]: Undefined attribute.
    if node.target.__name__ in node_visitors:  # type: ignore[union-attr]
        # pyre-ignore[16]: Undefined attribute.
        node_visitors[node.target.__name__].define_node(  # type: ignore[union-attr]
            node,
            tosa_graph,
            inputs,
            output,
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
    try:
        tosa_arg = TosaArg(node)
    except ValueError as e:
        raise ValueError(
            f"Failed processing input placeholder:\n{get_node_debug_info(node)}"
            "Is the original torch function supported?"
        ) from e
    input_shape = tosa_arg.shape
    input_dim_order = tosa_arg.dim_order
    tensor = ts.TosaSerializerTensor(
        tosa_arg.name,
        tosa_shape(input_shape, input_dim_order),
        tosa_arg.dtype,
        data=None,
        placeholderFilename=tosa_arg.name + ".npy",
    )
    tosa_graph.addInputTensor(tensor)


def process_inputs_to_parameters(
    node: torch.fx.Node,
    tosa_graph: ts.TosaSerializer,
    edge_program: ExportedProgram,
    tosa_spec: TosaSpecification,
):
    """Serialize bias and non-quantized weights"""
    try:
        tosa_arg = TosaArg(node)
    except ValueError as e:
        raise ValueError(
            f"Failed processing parameter placeholder:\n{get_node_debug_info(node)}"
            "Is the original torch function supported?"
        ) from e
    parameter_name = edge_program.graph_signature.inputs_to_parameters[tosa_arg.name]
    parameter_data = edge_program.state_dict[parameter_name]

    assert isinstance(parameter_data, torch.Tensor), "Expect Attr to be tensor"
    parameter_values = parameter_data.detach().numpy()

    if tosa_arg.dtype == torch.float32:
        assert tosa_spec.support_float(), f"{tosa_spec} doesn't support float"

    parameter_values = np.transpose(parameter_values, tosa_arg.dim_order)

    tosa_graph.addConst(
        parameter_values.shape, tosa_arg.dtype, parameter_values, name=tosa_arg.name
    )


def process_inputs_to_buffers(
    node: torch.fx.Node,
    tosa_graph: ts.TosaSerializer,
    edge_program: ExportedProgram,
):
    """Serialize quantized weights"""
    try:
        tosa_arg = TosaArg(node)
    except ValueError as e:
        raise ValueError(
            f"Failed processing buffer placeholder:\n{get_node_debug_info(node)}"
            "Is the original torch function supported?"
        ) from e
    buffer_name = edge_program.graph_signature.inputs_to_buffers[node.name]
    buffer_data = edge_program.state_dict[buffer_name]

    assert isinstance(buffer_data, torch.Tensor), "Expect Attr to be tensor"
    buffer_values = buffer_data.detach().numpy()

    # TODO: fragile code for temporary fix
    # the mean and var tensors are also stored here but they have shape (1, )
    # we only transpose weights here
    buffer_values = np.transpose(buffer_values, tosa_arg.dim_order)

    tosa_graph.addConst(
        buffer_values.shape, tosa_arg.dtype, buffer_values, name=node.name
    )


def process_inputs_to_lifted_tensor_constants(
    node: torch.fx.Node,
    tosa_graph: ts.TosaSerializer,
    edge_program: ExportedProgram,
):
    try:
        tosa_arg = TosaArg(node)
    except ValueError as e:
        raise ValueError(
            f"Failed processing lifted tensor constant placeholder:\n{get_node_debug_info(node)}"
            "Is the original torch function supported?"
        ) from e
    tensor_name = edge_program.graph_signature.inputs_to_lifted_tensor_constants[
        tosa_arg.name
    ]
    tensor = edge_program.tensor_constants[tensor_name]
    tensor_data = tensor.detach().numpy()

    tosa_graph.addConst(
        tensor_data.shape, tosa_arg.dtype, tensor_data, name=tosa_arg.name
    )


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
