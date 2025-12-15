# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import os
from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.devtools.visualization.visualization_utils import visualize_graph
from executorch.exir import ExportedProgram
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule


class VisualizePass(ArmPass):
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


class PrintGraphModuleCodePass(ArmPass):
    """
    This pass prints the graph module's code to stdout for debugging purposes.

    Example output:

      [arm_pass_manager.py:305]
      def forward(self, x, y):
          x, y, = fx_pytree.tree_flatten_spec(([x, y], {}), self._in_spec)
          remainder = torch.ops.aten.remainder.Scalar(x, 0.25);  x = None
          return pytree.tree_unflatten((remainder,), self._out_spec)
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def __init__(self, label: str | None = None):
        super().__init__()
        caller_frame = inspect.stack()[1]
        origin = f"{os.path.basename(caller_frame.filename)}:{caller_frame.lineno}"
        self.label = f"[{label}]" if label is not None else f"[{origin}]"

    def call(self, graph_module: GraphModule) -> PassResult:
        gm_code = graph_module.code.strip()
        print(f"\n{self.label}\n{gm_code}")
        return PassResult(graph_module, False)
