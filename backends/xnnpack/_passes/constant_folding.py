# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Mapping, Set
from typing import Any

import torch
from executorch.backends.transforms.utils import create_constant_placeholder
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind


def _evaluate_constants(
    graph: torch.fx.Graph,
    selected: Set[torch.fx.Node],
    known_values: Mapping[torch.fx.Node, Any],
) -> tuple[dict[torch.fx.Node, Any], list[torch.fx.Node]]:
    values = dict(known_values)
    evaluated = []
    with torch.no_grad():
        for node in graph.nodes:
            if node not in selected or not all(
                arg in values for arg in node.all_input_nodes
            ):
                continue
            args, kwargs = torch.fx.node.map_arg(
                (node.args, node.kwargs), values.__getitem__
            )
            value = node.target(*args, **kwargs)
            if isinstance(value, torch.Tensor):
                value = value.detach()
                if 0 in value.stride():
                    value = value.clone(memory_format=torch.contiguous_format)
            values[node] = value
            evaluated.append(node)
    return values, evaluated


def fold_constants(
    exported_program: ExportedProgram,
    selected: Set[torch.fx.Node],
    known_values: Mapping[torch.fx.Node, Any],
    kind: InputKind,
) -> bool:
    """Evaluate selected computations and lift tensors at surviving boundaries."""
    graph = exported_program.graph
    values, evaluated = _evaluate_constants(graph, selected, known_values)

    removable = set(evaluated)
    for node in reversed(evaluated):
        if not isinstance(values[node], torch.Tensor) and any(
            user not in removable for user in node.users
        ):
            removable.remove(node)

    taken = (
        {node.name for node in graph.nodes}
        | exported_program.state_dict.keys()
        | exported_program.constants.keys()
    )
    for node in evaluated:
        if node not in removable or all(user in removable for user in node.users):
            continue
        name = f"{node.name}_folded"
        while name in taken:
            name += "_"
        taken.add(name)
        with graph.inserting_before(next(iter(graph.nodes))):
            constant = create_constant_placeholder(
                exported_program, graph, name, kind, values[node]
            )
        # Keep fresh tensor metadata, including normalized strides.
        constant.meta = {**node.meta, **constant.meta}
        node.replace_all_uses_with(constant)

    for node in reversed(evaluated):
        if node in removable:
            graph.erase_node(node)
    if removable:
        exported_program.graph_module.recompile()
    return bool(removable)
