# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import torch
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchOpPackageInfo,
)

from .node_visitor import NodeVisitor
from .op_custom_op import CustomOp
from .utils import is_graph_input, is_graph_output, is_mutable_buffer_input


# This will hold mapping of all node names to the visitor class
_node_visitor_dict = {}


def register_node_visitor(visitor):
    """Register node visitor into _node_visitor_dict"""
    assert (
        isinstance(visitor, type)
        and issubclass(visitor, NodeVisitor)
        and hasattr(visitor, "target")
    ), f"Informed NodeVisitor subclass, can't register!, got: {visitor}"
    for target in visitor.target:
        _node_visitor_dict[target] = visitor


def generate_node_to_external_map(
    edge_program: torch.export.ExportedProgram,
) -> Dict[torch.fx.Node, int]:
    node_to_external_map = {}
    for node in edge_program.graph_module.graph.nodes:
        # The order in which we visit the placeholder node is same as the *args
        # order for the forward(*args) signature for this gm. Using the order of
        # the nodes as external_id to extract the right arg from *args at runtime
        if is_graph_input(node, edge_program) or is_mutable_buffer_input(
            node, edge_program
        ):
            node_to_external_map[node] = len(node_to_external_map)
    for node in edge_program.graph_module.graph.nodes:
        if is_graph_output(node):
            node_to_external_map[node] = len(node_to_external_map)
    return node_to_external_map


def get_node_visitors(
    edge_program: torch.export.ExportedProgram,
    enable_tensor_dump=False,
    op_package_infos: List[QnnExecuTorchOpPackageInfo] = None,
) -> Dict[str, NodeVisitor]:
    """Create a new class instance at runtime, and put them in a dict"""
    node_to_external_map = generate_node_to_external_map(edge_program)
    node_visitors = {}
    for target, visitor in _node_visitor_dict.items():
        assert callable(
            visitor
        ), f"Expecting a callable class, but got {visitor} of type {type(visitor)}"
        node_visitors[target] = visitor(
            node_to_external_map, edge_program, enable_tensor_dump
        )
    if op_package_infos:
        custom_ops = []
        for op_package_info in op_package_infos:
            if op_package_info.custom_op_name not in custom_ops:
                custom_op_builder = CustomOp(
                    op_package_info,
                    node_to_external_map,
                    edge_program,
                    enable_tensor_dump,
                )
                node_visitors[op_package_info.custom_op_name] = custom_op_builder
                custom_ops.append(op_package_info.custom_op_name)
    return node_visitors
