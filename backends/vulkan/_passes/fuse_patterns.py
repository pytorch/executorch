# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import executorch.backends.vulkan.patterns as vk_patterns

import torch

from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class FusePatternsPass(ExportPass):
    def __init__(self) -> None:
        super().__init__()
        self._exported_program: Optional[ExportedProgram] = None

    def call(self, graph_module: torch.fx.GraphModule):
        assert self._exported_program is not None

        total_replaced = vk_patterns.replace_all_fusable_subgraphs(
            self._exported_program, graph_module
        )

        if total_replaced > 0:
            for node in list(graph_module.graph.nodes):
                if node.target != exir_ops.edge.et_vk.select_as_symint.default:
                    continue
                value_range = node.meta.get("et_vk_value_range")
                if value_range is None:
                    continue
                lower_bound, upper_bound = value_range
                with graph_module.graph.inserting_after(node):
                    graph_module.graph.create_node(
                        "call_function",
                        exir_ops.edge.aten.sym_constrain_range.default,
                        args=(node,),
                        kwargs={"min": lower_bound, "max": upper_bound},
                    )
            graph_module.recompile()
            # Re-trace the graph
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, total_replaced > 0)
