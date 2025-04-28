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
    TensorArgument,
)


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

    # Create fake tensor using the same fake_mode as the other fake tensors in the graph
    example_node = list(graph.nodes)[0]
    if isinstance(
        example_node.meta["val"], (tuple, torch.fx.immutable_collections.immutable_list)
    ):
        example_fake_tensor = example_node.meta["val"][0]
    else:
        example_fake_tensor = example_node.meta["val"]
    fake_tensor = FakeTensorConverter().from_real_tensor(
        example_fake_tensor.fake_mode, t=data
    )

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
