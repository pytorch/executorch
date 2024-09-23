# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch
from torch._export.utils import (
    get_buffer,
    get_lifted_tensor_constant,
    get_param,
    is_buffer,
    is_lifted_tensor_constant,
    is_param,
)


def is_parameter(
    node: torch.fx.Node, edge_program: torch.export.ExportedProgram
) -> bool:
    return (
        is_param(edge_program, node)
        or is_buffer(edge_program, node)
        or is_lifted_tensor_constant(edge_program, node)
    )


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
    if param is not None:
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


def is_graph_output(tensor: torch.fx.Node) -> bool:
    """
    Check if the given tensor is used as a graph output

    Args:
        tensor: EdgeIR Tensor that is being checked for graph input
    """
    for user in tensor.users.keys():
        # getitem node is skiped, check the op_skip_ops.py
        if user.op == "output" or (
            user.target.__name__ == "getitem" and is_graph_output(user)
        ):
            return True
    return False


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
