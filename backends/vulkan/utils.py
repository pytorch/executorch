# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch._export.utils import is_buffer, is_param

from torch.export import ExportedProgram


def is_get_attr_node(node: torch.fx.Node) -> bool:
    return isinstance(node, torch.fx.Node) and node.op == "get_attr"


def is_constant(program: ExportedProgram, node: torch.fx.Node) -> bool:
    return node.name in program.graph_signature.inputs_to_lifted_tensor_constants


def is_param_node(program: ExportedProgram, node: torch.fx.Node) -> bool:
    """
    Check if the given node is a parameter within the exported program
    """
    return (
        is_get_attr_node(node)
        or is_param(program, node)
        or is_buffer(program, node)
        or is_constant(program, node)
    )
