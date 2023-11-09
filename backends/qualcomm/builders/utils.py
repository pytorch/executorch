# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


def is_parameter(
    node: torch.fx.Node, edge_program: torch.export.ExportedProgram
) -> bool:
    return (
        node.name in edge_program.graph_signature.inputs_to_parameters
        or node.name in edge_program.graph_signature.inputs_to_buffers
    )


def get_parameter(
    node: torch.fx.Node, edge_program: torch.export.ExportedProgram
) -> torch.Tensor:
    param = None
    if node.name in edge_program.graph_signature.inputs_to_parameters:
        param = edge_program.state_dict[
            edge_program.graph_signature.inputs_to_parameters[node.name]
        ].data
    if node.name in edge_program.graph_signature.inputs_to_buffers:
        param = edge_program.state_dict[
            edge_program.graph_signature.inputs_to_buffers[node.name]
        ]
    if param is not None:
        # update node.meta["val"] to qualified QNN datatype (e.g. i64 to i32)
        assert isinstance(param, torch.Tensor), "Expect parameter to be tensor"
        param = param.type(node.meta["val"].dtype)
    return param


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
        if user.op == "output":
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
