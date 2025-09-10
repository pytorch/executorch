# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.vulkan.patterns as vk_patterns

import torch

from executorch.exir import ExportedProgram
from executorch.exir.pass_base import ExportPass, PassResult


class FusePatternsPass(ExportPass):
    def __init__(self, exported_program: ExportedProgram) -> None:
        super().__init__()
        self.program = exported_program

    def call(self, graph_module: torch.fx.GraphModule):
        total_replaced = vk_patterns.replace_all_fusable_subgraphs(
            self.program, graph_module
        )

        if total_replaced > 0:
            graph_module.recompile()
            # Re-trace the graph
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, total_replaced > 0)
