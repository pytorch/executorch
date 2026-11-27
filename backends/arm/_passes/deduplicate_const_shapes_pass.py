# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node


class DeduplicateConstShapesPass(ArmPass):
    """Reuse the first CONST_SHAPE node with identical static values."""

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call(self, graph_module: GraphModule) -> PassResult:
        representatives: dict[tuple[int, ...], Node] = {}
        modified = False

        for node in list(graph_module.graph.nodes):
            if node.target != exir_ops.backend.tosa.CONST_SHAPE.default:
                continue

            values = node.args[0]
            if not isinstance(values, (list, tuple)) or not all(
                type(value) is int for value in values
            ):
                continue

            key = tuple(values)
            representative = representatives.get(key)
            if representative is None:
                representatives[key] = node
                continue

            node.replace_all_uses_with(representative)
            graph_module.graph.erase_node(node)
            modified = True

        if modified:
            graph_module.graph.lint()
            graph_module.recompile()

        return PassResult(graph_module, modified)
