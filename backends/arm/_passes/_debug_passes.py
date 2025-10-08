# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.devtools.visualization.visualization_utils import visualize_graph
from executorch.exir import ExportedProgram
from executorch.exir.pass_base import ExportPass, PassResult


class VisualizePass(ExportPass):
    """
    This pass visualizes the graph at the point of insertion in the pass manager
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def __init__(self, exported_program: ExportedProgram) -> None:
        super().__init__()
        self.exported_program = exported_program

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        visualize_graph(graph_module, self.exported_program)
        return PassResult(graph_module, False)
