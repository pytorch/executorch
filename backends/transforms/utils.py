# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from executorch.exir import ExportedProgram
from torch._export.utils import (
    get_buffer,
    get_lifted_tensor_constant,
    get_param,
    is_buffer,
    is_lifted_tensor_constant,
    is_param,
)
from torch._subclasses.fake_tensor import FakeTensorConverter
from torch.export.graph_signature import (
    ExportGraphSignature,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    TensorArgument,
)


def _get_fake_tensor_mode(graph: torch.fx.Graph, data: torch.Tensor) -> torch.Tensor:
    """
    Helper function to create a fake tensor using the fake_mode from existing nodes in the graph.

    Args:
        graph: The graph to get fake_mode from
        data: The tensor data to create fake tensor for

    Returns:
        A fake tensor with the appropriate fake_mode

    Raises:
        RuntimeError: If the graph has no nodes to extract fake_mode from
    """
    nodes = list(graph.nodes)
    if not nodes:
        raise RuntimeError(
            "Cannot create fake tensor: graph has no nodes to extract fake_mode from"
        )

    example_node = nodes[0]
    if isinstance(
        example_node.meta["val"], (tuple, torch.fx.immutable_collections.immutable_list)
    ):
        example_fake_tensor = example_node.meta["val"][0]
    else:
        example_fake_tensor = example_node.meta["val"]

    return FakeTensorConverter().from_real_tensor(example_fake_tensor.fake_mode, t=data)


def is_get_attr_node(node: torch.fx.Node) -> bool:
    """
    Returns true if the given node is a get attr node for a tensor of the model
    """
    return isinstance(node, torch.fx.Node) and node.op == "get_attr"


def is_param_node(exp_prog: ExportedProgram, node: torch.fx.Node) -> bool:
    return (
        is_get_attr_node(node)
        or is_param(exp_prog, node)
        or is_buffer(exp_prog, node)
        or is_lifted_tensor_constant(exp_prog, node)
    )


def get_param_tensor(
    exp_prog: ExportedProgram, node: torch.fx.Node
) -> Optional[torch.Tensor]:
    if node is None:
        return None
    elif is_param(exp_prog, node):
        return get_param(exp_prog, node)
    elif is_buffer(exp_prog, node):
        return get_buffer(exp_prog, node)
    elif is_lifted_tensor_constant(exp_prog, node):
        return get_lifted_tensor_constant(exp_prog, node)
    elif is_get_attr_node(node):
        # This is a hack to support both lifted and unlifted graph
        try:
            return getattr(node.graph.owning_module, node.target)
        except AttributeError:
            return getattr(exp_prog.graph_module, node.target)
    raise RuntimeError(f"unsupported param type, {node.op}.")


def create_constant_placeholder(
    exp_program: ExportedProgram,
    graph: torch.fx.Graph,
    name: str,
    kind: InputKind,
    data: torch.Tensor,
    persistent_buffer: Optional[bool] = None,
) -> torch.fx.Node:
    """
    Creates and returns a constant placeholder node, meaning that it is of type parameter, buffer,
    or lifted constant tensor. graph.inserting_before/after() should be used before the call to
    decide where to insert the node, at an insertion point before the first input node.
    """

    target = name

    # Add data to state_dict/ constants
    match kind:
        case InputKind.PARAMETER:
            exp_program.state_dict[target] = torch.nn.Parameter(
                data, requires_grad=False
            )
        case InputKind.BUFFER:
            if persistent_buffer is None:
                raise RuntimeError(
                    "Must set persistent_buffer when creating a new buffer."
                )
            elif persistent_buffer:
                exp_program.state_dict[target] = data
            else:
                exp_program.constants[target] = data
        case InputKind.CONSTANT_TENSOR:
            exp_program.constants[target] = data
        case _:
            raise RuntimeError("Can only create constant input nodes.")

    fake_tensor = _get_fake_tensor_mode(graph, data)

    # Create node
    node = graph.create_node(op="placeholder", name=name, target=name)
    node.meta["val"] = fake_tensor

    # Add tensor to graph_signature in the same order as nodes in the graph
    node_names = [n.name for n in graph.nodes if n.op == "placeholder"]
    node_index = node_names.index(name)

    input_specs = exp_program.graph_signature.input_specs
    user_input_indices = [
        i for i, spec in enumerate(input_specs) if spec.kind == InputKind.USER_INPUT
    ]
    if not all(
        (user_input_index >= node_index for user_input_index in user_input_indices)
    ):
        raise RuntimeError(
            f"Failed to insert {name}; Const placeholder nodes must be inserted before user input nodes in the graph."
        )

    arg_spec = TensorArgument(name)
    input_spec = InputSpec(kind, arg_spec, target, persistent_buffer)
    input_specs.insert(node_index, input_spec)

    new_graph_signature = ExportGraphSignature(
        input_specs, exp_program.graph_signature.output_specs
    )
    exp_program._graph_signature = new_graph_signature

    return node


def delete_constant_placeholder(exp_program: ExportedProgram, node: torch.fx.Node):
    """
    Deletes a node of type parameter, buffer, or lifted constant tensor and its related
    graph signature and state_dict/constant entries. The node may not have any users.
    """
    if not len(node.users) == 0:
        raise RuntimeError(
            f"Cannot delete input node {node.name} since it has users in the graph."
        )

    # Remove tensor from state_dict/ constants
    if node.name in exp_program.graph_signature.inputs_to_parameters:
        target = exp_program.graph_signature.inputs_to_parameters[node.name]
        del exp_program.state_dict[target]

    elif node.name in exp_program.graph_signature.inputs_to_buffers:
        target = exp_program.graph_signature.inputs_to_buffers[node.name]

        if target in exp_program.graph_signature.non_persistent_buffers:
            del exp_program.constants[target]
        else:
            del exp_program.state_dict[target]

    elif node.name in exp_program.graph_signature.inputs_to_lifted_tensor_constants:
        target = exp_program.graph_signature.inputs_to_lifted_tensor_constants[
            node.name
        ]
        del exp_program.constants[target]
    else:
        raise RuntimeError(
            f"Cannot delete input node {node.name} since it is not a parameter, a buffer, nor a lifted tensor constant."
        )

    # Remove input from graph signature
    input_specs = [
        spec
        for spec in exp_program.graph_signature.input_specs
        if spec.arg.name != node.name
    ]
    new_graph_signature = ExportGraphSignature(
        input_specs, exp_program.graph_signature.output_specs
    )
    exp_program._graph_signature = new_graph_signature

    # Remove node from graph
    node.graph.erase_node(node)


def _validate_graph_signature(exp_program: ExportedProgram):
    """
    Validates that the graph signature is up to date with the graph.
    """
    placeholders = [n for n in exp_program.graph.nodes if n.op == "placeholder"]
    if len(placeholders) != len(exp_program.graph_signature.input_specs):
        raise RuntimeError(
            f"Graph has {len(placeholders)} placeholder nodes but signature has "
            f"{len(exp_program.graph_signature.input_specs)} input specs"
        )
    for node, input_spec in zip(placeholders, exp_program.graph_signature.input_specs):
        if node.name != input_spec.arg.name:
            raise RuntimeError(
                f"Input node {node.name} does not match input spec {input_spec.arg.name}"
            )
    outputs = exp_program.graph.output_node().args[0]
    if len(outputs) != len(exp_program.graph_signature.output_specs):
        raise RuntimeError(
            f"Graph has {len(outputs)} output nodes but signature has "
            f"{len(exp_program.graph_signature.output_specs)} output specs"
        )
    for node, output_spec in zip(outputs, exp_program.graph_signature.output_specs):
        if node.name != output_spec.arg.name:
            raise RuntimeError(
                f"Output node {node.name} does not match output spec {output_spec.arg.name}"
            )


def _spec_to_node(
    exp_program: ExportedProgram, spec: InputSpec | OutputSpec
) -> torch.fx.Node:
    """
    Converts an InputSpec or OutputSpec to its corresponding node in the graph.
    """
    # Extract the argument name from the spec
    if hasattr(spec, "arg") and hasattr(spec.arg, "name"):
        arg_name = spec.arg.name
    else:
        raise RuntimeError(f"Invalid spec format: {spec}")

    # Find the corresponding node in the graph
    for node in exp_program.graph.nodes:
        if node.name == arg_name:
            return node

    raise RuntimeError(f"Could not find node with name '{arg_name}' in the graph")


def create_mutable_buffer(
    exp_program: ExportedProgram,
    name: str,
    data: torch.Tensor,
) -> torch.fx.Node:
    """
    Creates and returns a mutable buffer placeholder node. This is similar to
    create_constant_placeholder but specifically for creating mutable buffers that
    can be modified during execution.

    The difference between this and create_constant_placeholder is that this doesn't
    expect user to set the correct position for the placeholder node to be inserted,
    it finds the correct position automatically.

    It also updates the graph outputs to include the mutable buffer.

    Args:
        exp_program: The exported program to modify
        name: The name for the new buffer node (should start with "b_" prefix by convention)
        data: The initial tensor data for the buffer

    Returns:
        The created placeholder node to be used in the graph
    """
    # Input validation
    if not name or not name.strip():
        raise ValueError("Buffer name cannot be empty")

    if not isinstance(data, torch.Tensor):
        raise ValueError("Data must be a torch.Tensor")

    # Extract target name (remove "b_" prefix if present, following export convention)
    if name.startswith("b_"):
        target = name[2:]
    else:
        target = name

    # Check if target already exists
    if target in exp_program.state_dict:
        raise RuntimeError(f"Buffer target '{target}' already exists in state_dict")

    _validate_graph_signature(exp_program)

    persistent_buffer = True
    exp_program.state_dict[target] = data

    graph = exp_program.graph_module.graph

    # Create fake tensor using helper function
    fake_tensor = _get_fake_tensor_mode(graph, data)

    # Signature ordering is as follows:
    # Inputs = [*parameters_buffers_constant_tensors, *flattened_user_inputs]
    #                       ^^^^^^^
    #                       insert here (at the end of buffers)
    # Outputs = [*mutated_inputs, *flattened_user_outputs]
    #            ^^^^^^^^^^^^^^^
    #            insert here (at the end of mutated inputs)

    # Inputs
    # Find const or user input node if any, and insert before it
    node_index = 0
    node = None

    input_specs = exp_program.graph_signature.input_specs
    if len(input_specs) == 0 or all(
        spec.kind not in [InputKind.CONSTANT_TENSOR, InputKind.USER_INPUT]
        for spec in input_specs
    ):
        # No const or user input nodes
        node_index = len(input_specs)
        node = graph.create_node(op="placeholder", name=name, target=name)
    else:
        # Find the first constant or user input node
        for i, spec in enumerate(input_specs):
            if spec.kind in [InputKind.CONSTANT_TENSOR, InputKind.USER_INPUT]:
                node_index = i
                with graph.inserting_before(_spec_to_node(exp_program, spec)):
                    node = graph.create_node(op="placeholder", name=name, target=name)
                break

    assert node is not None, "node should be created at this point"
    node.meta["val"] = fake_tensor
    buffer_input_spec = InputSpec(
        InputKind.BUFFER, TensorArgument(name), target, persistent_buffer
    )
    input_specs.insert(node_index, buffer_input_spec)

    # Outputs
    # Create output spec for the mutable buffer, and insert it at the beginning of output specs
    user_output_indices = [
        i
        for i, spec in enumerate(exp_program.graph_signature.output_specs)
        if spec.kind == OutputKind.USER_OUTPUT
    ]

    output_index = user_output_indices[0] if user_output_indices else 0

    output_specs = exp_program.graph_signature.output_specs
    mutation_output_spec = OutputSpec(
        OutputKind.BUFFER_MUTATION, TensorArgument(name), target
    )
    output_specs.insert(output_index, mutation_output_spec)

    # Update the outputs to include the mutable buffer
    output_node = graph.output_node()
    args = list(output_node.args[0])
    args.insert(output_index, node)
    output_node.args = (args,)

    # Update graph signature in the exported program
    new_graph_signature = ExportGraphSignature(input_specs, output_specs)
    exp_program._graph_signature = new_graph_signature

    return node
