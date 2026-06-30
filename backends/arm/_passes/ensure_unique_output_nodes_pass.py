# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter
from typing import Any, Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class EnsureUniqueOutputNodesPass(ArmPass):
    """Ensure each graph output leaf references a unique producer node.

    If the same node appears multiple times in the output structure, insert a
    ``tosa.IDENTITY`` node for each occurrence and replace the repeated output
    entries with those identity nodes.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    @staticmethod
    def _collect_output_nodes(
        output_value: Any, counts: Counter[torch.fx.Node]
    ) -> None:
        if isinstance(output_value, torch.fx.Node):
            counts[output_value] += 1
            return
        if isinstance(output_value, (list, tuple)):
            for value in output_value:
                EnsureUniqueOutputNodesPass._collect_output_nodes(value, counts)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        output_node = graph.output_node()
        output_value = output_node.args[0]

        counts: Counter[torch.fx.Node] = Counter()
        self._collect_output_nodes(output_value, counts)
        repeated_nodes = {node for node, count in counts.items() if count > 1}
        if not repeated_nodes:
            return PassResult(graph_module, False)

        modified = False

        def _replace_repeated_outputs(value: Any) -> Any:
            nonlocal modified
            if isinstance(value, torch.fx.Node):
                if value not in repeated_nodes:
                    return value
                with graph.inserting_before(output_node):
                    identity_node = create_node(
                        graph,
                        exir_ops.backend.tosa.IDENTITY.default,
                        args=(value,),
                        from_node=value,
                    )
                modified = True
                return identity_node

            if isinstance(value, tuple):
                return tuple(_replace_repeated_outputs(v) for v in value)

            if isinstance(value, list):
                return [_replace_repeated_outputs(v) for v in value]

            return value

        new_output_value = _replace_repeated_outputs(output_value)
        if modified:
            output_node.args = (new_output_value,)
            graph.eliminate_dead_code()
            graph.lint()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
