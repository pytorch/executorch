# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.transforms import decompose_sdpa
from executorch.exir.pass_base import ExportPass, PassResult


class DecomposeScaledDotProductAttentionPass(
    ArmPass, decompose_sdpa.DecomposeScaledDotProductAttention
):
    """Pass that expands `aten.scaled_dot_product_attention` into primitive ops."""

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call(
        self, graph_module: torch.fx.GraphModule, allow_non_fake_inputs: bool = True
    ) -> PassResult:
        graph = graph_module.graph
        for node in list(graph.nodes):
            if node.target != torch.ops.aten.scaled_dot_product_attention.default:
                continue
            if not self.allowed_to_transform(node.meta):
                continue

            # Decompose with the superclass helper to reuse the shared logic.
            super()._decompose_sdpa_node(graph_module, node, allow_non_fake_inputs)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
