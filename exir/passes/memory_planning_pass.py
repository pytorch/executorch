# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import warnings
from typing import Callable, List, Optional

import torch
from executorch.exir.error import internal_assert
from executorch.exir.memory import alloc
from executorch.exir.memory_planning import (
    _is_out_var_node,
    apply_algo,
    get_node_tensor_specs,
    greedy,
    Verifier,
)
from executorch.exir.operator.convert import get_out_args_from_opoverload
from executorch.exir.pass_base import PassBase, PassResult
from executorch.exir.tensor import ALIGNMENT
from torch.export.exported_program import ExportGraphSignature


class MemoryPlanningPass(PassBase):
    def __init__(
        self,
        memory_planning_algo: Callable[..., List[int]] = greedy,
        allow_lifetime_and_storage_overlap: bool = False,
        alloc_graph_input: bool = True,
        alloc_graph_output: bool = True,
        alignment: int = ALIGNMENT,
    ) -> None:
        r"""
        alloc_graph_input/alloc_graph_output will have 4 different combinations
        to control if the memory planning algorithm need allocate memory for
        the graph input/output. The default behavior is the algorithm will allocate
        memory for both graph input and output.
        """
        self.memory_planning_algo = memory_planning_algo
        self.allow_lifetime_and_storage_overlap = allow_lifetime_and_storage_overlap
        self.alloc_graph_input = alloc_graph_input
        self.alloc_graph_output = alloc_graph_output
        self.alignment = alignment

    def _set_alloc_node_spec(self, graph_module: torch.fx.GraphModule) -> None:
        """
        Pass for setting all of the alloc node's specs. These nodes are created
        in the ToOutVarPass but do not have a spec.

        TODO(shunting): we probablly should setup the spec for memory.alloc node
          in the ToOutVarPass
        """
        for subgm in graph_module.modules():
            if not isinstance(subgm, torch.fx.GraphModule):
                continue
            for node in subgm.graph.nodes:
                if _is_out_var_node(node):
                    out_arg_names = get_out_args_from_opoverload(node.target)
                    if len(out_arg_names) == 1:
                        out_alloc_node = node.kwargs[out_arg_names[0]]
                        out_alloc_node.meta["spec"] = node.meta["spec"]
                        continue
                    specs = get_node_tensor_specs(node)
                    for i, out_arg in enumerate(out_arg_names):
                        out_alloc_node = node.kwargs[out_arg]
                        if out_alloc_node is None:
                            warnings.warn(
                                f"Function {node.target}'s {out_arg} kwarg value is None",
                                stacklevel=1,
                            )
                            continue
                        internal_assert(
                            out_alloc_node.op == "call_function"
                            and out_alloc_node.target == alloc,
                            f"Out-var's node {out_alloc_node} has op {out_alloc_node.op} and target {out_alloc_node.target}",
                        )
                        internal_assert(
                            "spec" not in out_alloc_node.meta,
                            f"Out-var's allocation node {out_alloc_node} already has a spec assigned",
                        )
                        out_alloc_node.meta["spec"] = specs[i]

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        return self.run(graph_module)

    def run(
        self,
        graph_module: torch.fx.GraphModule,
        graph_signature: Optional[ExportGraphSignature] = None,
    ) -> PassResult:
        """
        A pass for memory planning. The actual algorithm used will be picked by
        memory_planning_algo
        """
        self._set_alloc_node_spec(graph_module)
        # TODO(shunting) if people have concern of adding a field to GraphModule
        # directly, we should define a GraphModule subclass that we can add our
        # customized fields. Using the graph_module object to convey information across
        # passes/stages is quite natural and avoid yet another 'context' data structure
        # to do the job.
        _ = apply_algo(
            self.memory_planning_algo,
            graph_module,
            self.alignment,
            graph_signature,
            self.alloc_graph_input,
            self.alloc_graph_output,
        )

        # TODO: make the verifier do the work recursively to handle
        # control flow
        verifier = Verifier(
            graph_module,
            self.alloc_graph_input,
            self.alloc_graph_output,
            graph_signature,
        )

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            num_reuse_pairs = verifier.verify_storage_reuse(
                self.allow_lifetime_and_storage_overlap
            )
            logging.debug(
                f"The {getattr(self.memory_planning_algo, '__name__', repr(self.memory_planning_algo))} algorithm reuses storage for {num_reuse_pairs} pair of tensors"
            )
        verifier.verify_graph_input_output()
        return PassResult(graph_module, True)
