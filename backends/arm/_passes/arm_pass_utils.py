# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from inspect import isclass
from typing import Optional

import torch
import torch.fx

from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops

from torch._export.utils import (
    get_buffer,
    get_lifted_tensor_constant,
    get_param,
    is_buffer,
    is_lifted_tensor_constant,
    is_param,
)
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorConverter
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
            return getattr(node.graph.owning_module, node.target)  # type: ignore[arg-type]
        except AttributeError:
            return getattr(exp_prog.graph_module, node.target)  # type: ignore[arg-type]
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
    Creates and returns a constant placeholder node, meaning that it is of type parameter, buffer, or lifted constant tensor.
    graph.inserting_before/after() should be used before the call to decide where to insert the node.
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

    # Create node
    fake_tensor_mode = get_first_fake_tensor(
        list(graph.nodes)[0]
    ).fake_mode  # Use the same fake_tensor_mode as all other fake tensors in the graph
    node = graph.create_node(op="placeholder", name=name, target=name)
    node.meta["val"] = FakeTensorConverter().from_real_tensor(fake_tensor_mode, t=data)

    # Add tensor to graph_signature in the same order as nodes in the graph
    node_names = [n.name for n in graph.nodes if n.op == "placeholder"]
    node_index = node_names.index(name)

    input_specs = exp_program.graph_signature.input_specs
    user_input_indices = [
        i for i, spec in enumerate(input_specs) if spec.kind == InputKind.USER_INPUT
    ]
    if not all(
        (user_input_index > node_index for user_input_index in user_input_indices)
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
    Deletes a constant placeholder node, meaning that it is of type parameter, buffer, or lifted constant tensor,
    if the node does not have any users.
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


def create_node(
    graph: torch.fx.Graph,
    op_target: OpOverload,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    quantize: bool = False,
    q_params: Optional[tuple] = None,
):
    """
    Adds a node to 'graph'. graph.inserting_before/after() should be used before the call to decide where to insert the node.
    If quantize is true and q_params is not None, a q dq pair is inserted after the newly created node.
    """

    node = graph.create_node(
        "call_function",
        op_target,
        args=args,
        kwargs=kwargs or {},
    )
    if quantize and q_params:
        return insert_q_dq_pair(graph, node, q_params)
    return node


def insert_q_dq_pair(
    graph: torch.fx.Graph,
    anchor: torch.fx.Node,
    q_params: tuple,
):
    """
    Inserts a q dq node pair after the node 'anchor'.
    """

    with graph.inserting_after(anchor):
        q = create_node(
            graph=graph,
            op_target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(),  # We add the argument last
        )
        q.meta = anchor.meta
    with graph.inserting_after(q):
        dq = create_node(
            graph=graph,
            op_target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(q,) + q_params,
        )
        dq.meta = q.meta
    anchor.replace_all_uses_with(dq)
    # We add this last so the replace all uses above does not replace the quantized
    # node's first use
    q.args = (anchor,) + q_params
    return dq


def get_first_fake_tensor(node: torch.fx.Node) -> FakeTensor:
    """
    Returns a FakeTensor from the meta field of 'node'.
    If the node contains many fake tensors, return the first one.
    """
    if isinstance(
        node.meta["val"], (tuple, torch.fx.immutable_collections.immutable_list)
    ):
        fake_tensor = node.meta["val"][0]
    else:
        fake_tensor = node.meta["val"]

    assert isinstance(
        fake_tensor, FakeTensor
    ), f'Found {fake_tensor} in meta["val"] of {node}, expected to find FakeTensor.'
    return fake_tensor


def get_node_arg(args: list | dict, key: int | str | type, default_value=None):
    """
    Help-function for getting a value from node.args/ kwargs, three cases:
    1. By position in node.args - Returns arg at given position or default_value if index is one out of bounds
    2. By key in node.kwargs - Returns kwarg with given key or default_value if it deos not exist
    3. By type in node.args - Returns first arg of args of given type. Useful for cases where arg postions may differ but types are unique.
    """
    if isinstance(key, int):
        if 0 <= key < len(args):
            return args[key]
        elif key == len(args):
            if default_value is not None:
                return default_value
            else:
                raise RuntimeError(f"No defult value given for index {key}")
        else:
            raise RuntimeError(
                f"Out of bounds index {key} for getting value in args (of size {len(args)})"
            )
    elif isinstance(key, str):
        return args.get(key, default_value)  # type: ignore[union-attr]  # pyre-ignore[16]
    elif isclass(key):
        for arg in args:
            if isinstance(arg, key):
                return arg
        if default_value is not None:
            return default_value
        else:
            raise RuntimeError(f"No arg of type {key}")
    else:
        raise RuntimeError("Invalid type")


def set_node_arg(node: torch.fx.Node, i: int | str, value):
    """
    Help-function for setting a value in node.args/ kwargs. If the index is one larger than the list size, the value is instead appended to the list.
    """
    if isinstance(i, int):
        if 0 <= i < len(node.args):
            args = list(node.args)
            args[i] = value
            node.args = tuple(args)
            return
        elif i == len(node.args):
            node.args = node.args + (value,)
        else:
            raise RuntimeError(
                f"Out of bounds index {i} for setting value in {node} args (of size {len(node.args)})"
            )
    elif isinstance(i, str):
        kwargs = dict(node.kwargs)
        kwargs[i] = value
        node.kwargs = kwargs
    else:
        raise RuntimeError("Invalid type")
