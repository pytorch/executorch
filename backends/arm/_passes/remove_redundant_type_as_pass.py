# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch

from executorch.backends.arm._passes import ArmPass
from executorch.exir.pass_base import ExportPass, PassResult


class RemoveRedundantTypeAsPass(ArmPass):
    """Remove no-op type_as calls on parameters and buffers."""

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        for node in list(graph_module.graph.nodes):
            if (
                node.op != "call_function"
                or node.target != torch.ops.aten.type_as.default
            ):
                continue

            source = node.args[0]
            if not isinstance(source, torch.fx.Node) or source.op != "get_attr":
                continue

            source_value = source.meta.get("val")
            output_value = node.meta.get("val")
            if not isinstance(source_value, torch.Tensor) or not isinstance(
                output_value, torch.Tensor
            ):
                continue
            if (
                source_value.dtype != output_value.dtype
                or source_value.device != output_value.device
            ):
                continue

            node.replace_all_uses_with(source)
            graph_module.graph.erase_node(node)
            modified = True

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.graph.lint()
            graph_module.recompile()

        return PassResult(graph_module, modified)
