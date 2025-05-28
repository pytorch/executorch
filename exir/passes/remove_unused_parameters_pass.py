# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch

from torch.export.exported_program import ExportedProgram, InputKind


def remove_unused_parameters_pass(
    ep: ExportedProgram,
) -> ExportedProgram:
    """
    Remove unused parameters from the exported program.
    """

    placeholder_nodes = {
        node.target: node
        for node in ep.graph_module.graph.nodes
        if node.op == "placeholder"
    }

    unused_parameters = [
        s
        for s in ep.graph_signature.input_specs
        if s.kind == InputKind.PARAMETER
        and not _is_parameter_used(ep, s.arg.name, placeholder_nodes)
    ]

    # Remove params from the state dict, graph, and signature.
    new_signature = copy.deepcopy(ep.graph_signature)
    for param in unused_parameters:
        new_signature.input_specs.remove(param)
        del ep._state_dict[param.target]
        ep.graph_module.graph.erase_node(placeholder_nodes[param.arg.name])

    ep._graph_signature = new_signature
    ep.graph_module.recompile()
    return ep


def _is_parameter_used(
    ep: ExportedProgram, parameter: str, placeholder_nodes: dict[str, torch.fx.Node]
) -> bool:
    placeholder_node = placeholder_nodes.get(parameter)
    if placeholder_node is None:
        raise RuntimeError(
            f"Invalid graph. No placeholder for {parameter} found in graph."
        )

    return len(placeholder_node.users) > 0
