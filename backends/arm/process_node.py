# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

import operator
from typing import Any, cast, Dict

import numpy as np
import torch
import torch.fx
import tosa_serializer as ts
from executorch.backends.arm.operators.node_visitor import NodeVisitor
from executorch.backends.arm.tosa.mapping import TosaArg, TosaSpecialDtype
from executorch.backends.arm.tosa.specification import TosaSpecification
from executorch.backends.arm.tosa.utils import tosa_shape
from torch._export.utils import (
    get_buffer,
    get_lifted_tensor_constant,
    get_param,
    is_buffer,
    is_lifted_tensor_constant,
    is_param,
)
from torch.export.exported_program import ExportedProgram


def process_call_function(
    node: torch.fx.Node,
    tosa_graph: Any,
    node_visitors: Dict[str, NodeVisitor],
    tosa_spec: TosaSpecification,
):
    # Unpack arguments and convert
    try:
        inputs = [TosaArg(arg, tosa_spec) for arg in node.args]
    except ValueError as e:
        raise ValueError(f"Failed processing args to op:\n{node}") from e

    # Convert output (this node itself)
    try:
        output = TosaArg(node, tosa_spec)
    except ValueError as e:
        raise ValueError(
            f"Failed processing call_function: {node.name}. "
            "Is the original torch function supported?"
        ) from e

    if not output.multiple_output_names:
        tosa_graph.currRegion.currBasicBlock.addTensor(
            output.name, tosa_shape(output.shape, output.dim_order), output.dtype
        )

    # Get item nodes just add tensors, no node visitor is needed.
    if node.target == operator.getitem:
        return

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
    tosa_graph: Any,
    tosa_spec: TosaSpecification,
):
    """Serialize an input node"""
    try:
        tosa_arg = TosaArg(node, tosa_spec)
    except ValueError as e:
        raise ValueError(
            f"Failed processing input placeholder: {node.name}. "
            "Is the original torch function supported?"
        ) from e

    input_shape = tosa_arg.shape
    input_dim_order = tosa_arg.dim_order
    tensor = ts.TosaSerializerTensor(
        tosa_arg.name,
        tosa_shape(input_shape, input_dim_order),
        tosa_arg.dtype,
        data=None,
    )
    tosa_graph.addInputTensor(tensor)


def process_inputs_to_parameters(
    node: torch.fx.Node,
    tosa_graph: Any,
    edge_program: ExportedProgram,
    tosa_spec: TosaSpecification,
):
    """Serialize bias and non-quantized weights"""
    try:
        tosa_arg = TosaArg(node, tosa_spec)
    except ValueError as e:
        raise ValueError(
            f"Failed processing parameter placeholder: {node.name}. "
            "Is the original torch function supported?"
        ) from e
    parameter_data = get_param(edge_program, node)

    if not isinstance(parameter_data, torch.Tensor):
        raise TypeError(
            f"Expected parameter '{node.name}' to be a torch.Tensor, got "
            f"{type(parameter_data).__name__}"
        )
    parameter_values = parameter_data.detach().numpy()

    if tosa_arg.dtype == torch.float32:
        if not tosa_spec.support_float():
            raise ValueError(f"{tosa_spec} doesn't support float operations")

    # Handle special case for INT48 tensors
    special_type = node.meta.get(TosaSpecialDtype.meta_key(), None)
    if isinstance(special_type, TosaSpecialDtype):
        tosa_dtype = special_type.get_tosa_dtype()
    else:
        tosa_dtype = tosa_arg.dtype

    parameter_values = np.transpose(parameter_values, tosa_arg.dim_order)

    tosa_graph.addConst(
        parameter_values.shape, tosa_dtype, parameter_values, name=tosa_arg.name
    )


def process_inputs_to_buffers(
    node: torch.fx.Node,
    tosa_graph: Any,
    edge_program: ExportedProgram,
    tosa_spec: TosaSpecification,
):
    """Serialize quantized weights"""
    try:
        tosa_arg = TosaArg(node, tosa_spec)
    except ValueError as e:
        raise ValueError(
            f"Failed processing buffer placeholder: {node.name}. "
            "Is the original torch function supported?"
        ) from e
    buffer_data = get_buffer(edge_program, node)

    if not isinstance(buffer_data, torch.Tensor):
        raise TypeError(
            f"Expected buffer '{node.name}' to be a torch.Tensor, got "
            f"{type(buffer_data).__name__}"
        )
    buffer_values = buffer_data.detach().numpy()

    # TODO: fragile code for temporary fix
    # the mean and var tensors are also stored here but they have shape (1, )
    # we only transpose weights here
    buffer_values = np.transpose(buffer_values, tosa_arg.dim_order)

    tosa_graph.addConst(
        buffer_values.shape, tosa_arg.dtype, buffer_values, name=tosa_arg.name
    )


def process_inputs_to_lifted_tensor_constants(
    node: torch.fx.Node,
    tosa_graph: Any,
    edge_program: ExportedProgram,
    tosa_spec: TosaSpecification,
):
    try:
        tosa_arg = TosaArg(node, tosa_spec)
    except ValueError as e:
        raise ValueError(
            f"Failed processing lifted tensor constant placeholder: {node.name}. "
            "Is the original torch function supported?"
        ) from e
    tensor = get_lifted_tensor_constant(edge_program, node)
    tensor_data = tensor.detach().numpy()  # type: ignore[union-attr]

    tosa_graph.addConst(
        tensor_data.shape, tosa_arg.dtype, tensor_data, name=tosa_arg.name
    )


def process_placeholder(
    node: torch.fx.Node,
    tosa_graph: Any,
    edge_program: ExportedProgram,
    tosa_spec: TosaSpecification,
):
    """Wrapper for processing and serializing all types of placeholders"""
    if node.name != node.target:
        raise ValueError(
            f"Placeholder name '{node.name}' does not match target '{node.target}'"
        )
    if len(node.args) != 0:
        raise ValueError(f"Placeholder '{node.name}' must not have default values")

    if node.name in edge_program.graph_signature.user_inputs:
        process_inputs(node, tosa_graph, tosa_spec)
    elif is_param(edge_program, node):
        process_inputs_to_parameters(node, tosa_graph, edge_program, tosa_spec)
    elif is_buffer(edge_program, node):
        process_inputs_to_buffers(node, tosa_graph, edge_program, tosa_spec)
    elif is_lifted_tensor_constant(edge_program, node):
        process_inputs_to_lifted_tensor_constants(
            node, tosa_graph, edge_program, tosa_spec
        )
    elif node.name in edge_program.graph_signature.inputs_to_lifted_custom_objs:
        raise NotImplementedError(
            "Placeholder is of type 'lifted custom object' which is not supported."
        )
    else:
        raise RuntimeError(f"Placeholder '{node.name}' is of unknown type.")


def process_output(node: torch.fx.Node, tosa_graph: Any, tosa_spec: TosaSpecification):
    for output in cast(tuple[torch.fx.Node, ...], node.args[0]):
        output_arg = TosaArg(output, tosa_spec)
        tosa_graph.addOutputTensor(
            tosa_graph.currRegion.currBasicBlock.tensors[output_arg.name]
        )
