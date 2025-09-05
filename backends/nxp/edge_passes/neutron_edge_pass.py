# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from abc import abstractmethod

import torch

from executorch.exir.pass_base import ExportPass
from torch.fx.passes.infra.pass_base import PassResult


class NeutronEdgePass(ExportPass):
    """Abstract parent class for pre-processing passes on the edge dialect level."""

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        """Call `self.run()` as long as changes are being made. After a pass modifies the graph, it cannot keep on
        iterating through its nodes, and must return. This method allows the pass to go through the whole model.
        """

        # Every pass will return once it makes a change to the graph, to avoid traversing and modifying a graph at the
        #  same time. Therefore, it must be called multiple times (at most `iteration_limit` times).
        iteration_limit = len(graph_module.graph.nodes)
        modified = False
        for _ in range(iteration_limit):
            res = self.run(graph_module)
            if res.modified:
                modified = True
                graph_module = res.graph_module

            else:
                # No more changes have been made.
                graph_module = self.recompile_module(graph_module)
                return PassResult(graph_module, modified)

        # Iteration limit was reached.
        logging.warning(
            f"The NeutronEdgePass `{self.__class__.__name__}` reached the iteration limit."
        )
        graph_module = self.recompile_module(graph_module)
        return PassResult(graph_module, modified)

    @abstractmethod
    def run(self, graph_module: torch.fx.GraphModule) -> PassResult:
        """Child classes should implement their graph modification here."""
        pass

    def recompile_module(
        self, graph_module: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        """Recompile the graph and re-trace the metadata. This should ensure that the datatypes and shapes are correct."""
        graph_module.recompile()
        return super().call(graph_module).graph_module
