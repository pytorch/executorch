# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import Any, Dict, Optional

import torch
from executorch.backends.qualcomm.utils.constants import (
    QCOM_CUSTOM_OP_INPUT_AXIS_ORDERS,
    QCOM_CUSTOM_OP_OUTPUT_AXIS_ORDERS,
    QCOM_FILTER_AXIS_ORDER_HWIO,
    QCOM_FILTER_AXIS_ORDER_HWOI,
    QCOM_IS_RUNTIME_INPUT,
    QCOM_RUNTIME_INPUTS,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch._export.utils import (
    get_buffer,
    get_lifted_tensor_constant,
    get_param,
    is_buffer,
    is_lifted_tensor_constant,
    is_param,
)


def get_attr_from_target(graph_module: torch.fx.GraphModule, target: str) -> Any:
    """Resolve a possibly dotted get_attr target (e.g. ``linear.weight``)."""
    attr: Any = graph_module
    for target_atom in target.split("."):
        attr = getattr(attr, target_atom)
    return attr


def set_attr_from_target(
    graph_module: torch.fx.GraphModule, target: str, replacement: Any
) -> None:
    """Assign to a possibly dotted get_attr target (e.g. ``linear.weight``)."""
    attr: Any = graph_module
    target_list = target.split(".")
    for target_atom in target_list[:-1]:
        attr = getattr(attr, target_atom)
    setattr(attr, target_list[-1], replacement)


def _argument_contains_node(argument: Any, node: torch.fx.Node) -> bool:
    if argument is node:
        return True
    if isinstance(argument, (list, tuple)):
        return any(_argument_contains_node(value, node) for value in argument)
    if isinstance(argument, dict):
        return any(_argument_contains_node(value, node) for value in argument.values())
    return False


def _runtime_input_indices(graph: torch.fx.Graph) -> dict[torch.fx.Node, set[int]]:
    indices = {}
    for node in graph.nodes:
        indices[node] = set(node.meta.get(QCOM_RUNTIME_INPUTS, ()))
        if any(index < 0 or index >= len(node.args) for index in indices[node]):
            raise ValueError(f"Runtime input index for {node.name} is out of range.")

    # A cast does not make an immutable runtime buffer a compile-time constant.
    cast_targets = {
        exir_ops.edge.aten._to_copy.default,
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
    }
    for node in reversed(list(graph.nodes)):
        if node.target in cast_targets and any(
            _argument_contains_node(user.args[index], node)
            for user in node.users
            for index in indices[user]
        ):
            indices[node].add(0)
    return indices


def mark_runtime_inputs(edge_program: torch.export.ExportedProgram) -> None:
    """Validate QCOM_RUNTIME_INPUTS and cache runtime-input classification."""
    graph_signature = edge_program.graph_signature
    runtime_input_indices = _runtime_input_indices(edge_program.graph)
    for node in edge_program.graph.nodes:
        node.meta.pop(QCOM_IS_RUNTIME_INPUT, None)
        user_indices = {}
        for user in node.users:
            runtime_indices = runtime_input_indices[user]
            indices = {
                index
                for index, argument in enumerate(user.args)
                if _argument_contains_node(argument, node)
            }
            if indices:
                user_indices[user] = (indices, runtime_indices)

        if not any(
            indices & runtime_indices
            for indices, runtime_indices in user_indices.values()
        ):
            continue
        buffer_name = graph_signature.inputs_to_buffers.get(node.name)
        if buffer_name in graph_signature.buffers_to_mutate.values():
            raise ValueError(f"Runtime input {node.name} must not be a mutable buffer.")
        if is_parameter(node, edge_program) and any(
            not indices <= runtime_indices
            for indices, runtime_indices in user_indices.values()
        ):
            raise ValueError(
                f"Runtime input {node.name} must not have non-runtime users."
            )
        node.meta[QCOM_IS_RUNTIME_INPUT] = True


def is_runtime_input(node: torch.fx.Node) -> bool:
    return node.meta.get(QCOM_IS_RUNTIME_INPUT, False)


def get_custom_op_output_axis_order(
    node: torch.fx.Node,
) -> Optional[tuple[int, ...]]:
    producer, output_index = get_custom_op_output_source(node)
    return producer.meta.get(QCOM_CUSTOM_OP_OUTPUT_AXIS_ORDERS, {}).get(output_index)


def get_custom_op_output_producer(node: torch.fx.Node) -> torch.fx.Node:
    if node.target is operator.getitem and isinstance(node.args[0], torch.fx.Node):
        return node.args[0]
    return node


def get_custom_op_output_source(node: torch.fx.Node) -> tuple[torch.fx.Node, int]:
    producer = get_custom_op_output_producer(node)
    if producer is not node and isinstance(node.args[1], int):
        return producer, node.args[1]
    return node, 0


def uses_compatible_axis_order(
    user: torch.fx.Node, input_node: torch.fx.Node, axis_order: tuple[int, ...]
) -> bool:
    input_indices = [
        index for index, argument in enumerate(user.args) if argument is input_node
    ]
    if user.target == exir_ops.edge.aten.convolution.default:
        return (
            input_indices == [1]
            and not user.args[6]
            and axis_order == QCOM_FILTER_AXIS_ORDER_HWIO
        )
    if user.target == exir_ops.edge.aten.convolution_backward.default:
        return input_indices == [2] and axis_order in (
            QCOM_FILTER_AXIS_ORDER_HWIO,
            QCOM_FILTER_AXIS_ORDER_HWOI,
        )
    input_axis_orders = user.meta.get(QCOM_CUSTOM_OP_INPUT_AXIS_ORDERS, {})
    return bool(input_indices) and all(
        input_axis_orders.get(index) == axis_order for index in input_indices
    )


def is_parameter(
    node: torch.fx.Node, edge_program: torch.export.ExportedProgram
) -> bool:
    return (
        node.op == "get_attr"
        or is_param(edge_program, node)
        or is_buffer(edge_program, node)
        or is_lifted_tensor_constant(edge_program, node)
    ) and not is_runtime_input(node)


def get_parameter(
    node: torch.fx.Node, edge_program: torch.export.ExportedProgram
) -> torch.Tensor:
    param = None
    if is_param(edge_program, node):
        param = get_param(edge_program, node)
    if is_buffer(edge_program, node):
        param = get_buffer(edge_program, node)
    if is_lifted_tensor_constant(edge_program, node):
        param = get_lifted_tensor_constant(edge_program, node)
    if node.op == "get_attr":
        assert isinstance(node.target, str)
        try:
            param = getattr(node.graph.owning_module, node.target)
        except AttributeError:
            param = getattr(edge_program.graph_module, node.target)
    assert (
        param is not None
    ), f"Expect {node.name} to be parameter, buffer, get_attr, or lifted tensor constant"
    # update node.meta["val"] to qualified QNN datatype (e.g. i64 to i32)
    assert isinstance(param, torch.Tensor), "Expect parameter to be tensor"
    param = param.type(node.meta["val"].dtype)
    return param


def set_parameter(
    param: torch.Tensor, node: torch.fx.Node, edge_program: torch.export.ExportedProgram
):
    status = False
    if is_param(edge_program, node):
        edge_program.state_dict[
            edge_program.graph_signature.inputs_to_parameters[node.name]
        ] = param
        status = True
    if is_buffer(edge_program, node):
        buffer_name = edge_program.graph_signature.inputs_to_buffers[node.name]
        if buffer_name in edge_program.graph_signature.non_persistent_buffers:
            edge_program.constants[buffer_name] = param
        else:
            edge_program.state_dict[buffer_name] = param
        status = True
    assert status, "Failed to set parameter"


def is_graph_input(
    tensor: torch.fx.Node, edge_program: torch.export.ExportedProgram
) -> bool:
    """
    Check if the given tensor is a graph input

    Args:
        tensor: EdgeIR Tensor that is being checked for graph input
    """
    return tensor.op == "placeholder" and not is_parameter(tensor, edge_program)


def is_mutable_buffer_input(
    tensor: torch.fx.Node, edge_program: torch.export.ExportedProgram
) -> bool:
    """
    Check if the given tensor is a mutable buffer input
    Args:
        tensor: EdgeIR Tensor that is being checked for mutable buffer input
    """
    if tensor.op == "placeholder" and is_buffer(edge_program, tensor):
        fqn = edge_program.graph_signature.inputs_to_buffers[tensor.target]
        # if the buffer is mutated then record that
        return fqn in edge_program.graph_signature.buffers_to_mutate.values()


def is_graph_output(node: torch.fx.Node, output_index: Optional[int] = None) -> bool:
    """
    Check if the given tensor is used as a graph output

    Args:
        node: EdgeIR Tensor that is being checked for graph output
        output_index: for a multi-output node, restrict the check to this
            output. Without it a single escaping output would mark every
            output of the node as a graph output, publishing values that
            nothing consumes.
    """
    for user in node.users.keys():
        if user.op == "output":
            return True
        # getitem node is skipped, check the op_skip_ops.py
        # call_module targets are plain strings and have no __name__
        if getattr(user.target, "__name__", "") == "getitem":
            if output_index is not None and user.args[1] != output_index:
                continue
            if is_graph_output(user):
                return True
    return False


def is_mutable_buffer_output(
    tensor: torch.fx.Node, edge_program: torch.export.ExportedProgram
) -> bool:
    """
    Check if the given tensor is a mutable buffer output
    Args:
        tensor: EdgeIR Tensor that is being checked for mutable buffer output
    """
    return (
        any(
            user.op == "output"
            or getattr(user.target, "__name__", "") == "getitem"
            and is_graph_output(user)
            for user in tensor.users.keys()
        )
        and tensor.name in edge_program.graph_signature.buffers_to_mutate.keys()
    )


def is_constant(
    tensor: torch.fx.Node, edge_program: torch.export.ExportedProgram
) -> bool:
    """
    Check if the given tensor is a constant

    Args:
        tensor: EdgeIR Tensor that is being checked for graph input
    """
    # constants should not be treated as input placeholder
    # pay attention to the pytorch design, change this if
    # breakage happened:
    # pytorch/torch/_export/passes/lift_constant_tensor_pass.py
    if is_parameter(tensor, edge_program):
        return tensor.meta["val"].constant is not None

    return False


def deduce_dtype(
    tensor: torch.Tensor, quant_infos: Optional[Dict] = None
) -> torch.dtype:
    if quant_infos:
        quant_range = quant_infos["quant_max"] - quant_infos["quant_min"]
        unsigned = quant_infos["quant_min"] >= 0
        if quant_range <= torch.iinfo(torch.int8).max - torch.iinfo(torch.int8).min:
            return torch.uint8 if unsigned else torch.int8

        elif quant_range <= torch.iinfo(torch.int16).max - torch.iinfo(torch.int16).min:
            return torch.uint16 if unsigned else torch.int16

        return quant_infos["dtype"]

    return tensor.dtype
