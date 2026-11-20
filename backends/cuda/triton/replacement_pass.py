# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Graph Transformation Pass for Triton Kernel Replacement.

This pass replaces ATen operators with optimized Triton kernels in the graph.
"""

import logging
from typing import Callable

from executorch.backends.cuda.triton import measured_routing, routing
from executorch.backends.cuda.triton.kernels.conv import routing as conv_routing
from executorch.backends.cuda.triton.kernels.sdpa import plan as plan_sdpa
from executorch.backends.cuda.triton.kernels.topk import plan as plan_topk
from executorch.exir.dialects._ops import ops as exir_ops

from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult

logger = logging.getLogger(__name__)

# The ops the pass replaces: edge op -> planner(node) returning a
# ``routing.ReplacementPlan``, which lists the implementations of the call
# (see routing.py).
PLANNERS: dict[object, Callable[[Node], routing.ReplacementPlan]] = {
    exir_ops.edge.aten.scaled_dot_product_attention.default: plan_sdpa,
    exir_ops.edge.aten.topk.default: plan_topk,
    **{target: conv_routing.plan for target in conv_routing.CONV_TARGETS},
}


class ReplaceEdgeOpWithTritonOpPass(PassBase):
    """
    Pass to replace ATen operators with Triton kernels.

    This pass plans every call of an op in ``PLANNERS`` and replaces it with
    the implementation its plan chooses.
    """

    def __init__(self, measured_routing: bool = False):
        """Initialize the pass.

        Args:
            measured_routing: Leave a planned call that several
                implementations can run, of a family that opts in, to
                ``MeasureChoicesPass`` and ``ApplyChoicesPass``; otherwise take
                its plan's static choice.
        """
        super().__init__()
        self._measured_routing = measured_routing
        self._replacement_count = 0
        self._plans: dict[Node, tuple[routing.ReplacementPlan, routing.Candidate]] = {}

    def call(self, graph_module: GraphModule) -> PassResult:
        """
        Execute the pass on the graph module.

        Args:
            graph_module: The graph module to transform

        Returns:
            PassResult indicating success/failure and the modified graph module
        """
        self._replacement_count = 0
        self._plans = {}
        modified = False

        for node in list(graph_module.graph.nodes):
            if not self._should_replace_node(node):
                continue
            # Rolls back on failure and raises only for calls that must be
            # replaced.
            replaced = self._replace_node_with_triton(graph_module, node)
            if replaced:
                modified = True
                self._replacement_count += 1

        if modified:
            # Recompile the graph module after modifications
            graph_module.recompile()

        logger.info(f"Replaced {self._replacement_count} nodes with Triton kernels")

        return PassResult(graph_module, modified)

    def _should_replace_node(self, node: Node) -> bool:
        """
        Check if a node should be replaced with a Triton kernel.

        Args:
            node: The node to check

        Returns:
            True if the node should be replaced
        """
        if node.op != "call_function" or node.target not in PLANNERS:
            return False

        plan = PLANNERS[node.target](node)
        legal = plan.legal()
        if not legal:
            raise routing.ReplacementError(
                f"{plan.signature}: nothing can run it ({plan.describe()})"
            )
        if len(legal) > 1 and self._measured_routing and plan.measure:
            measured_routing.record_plan(node, plan)
            logger.info(f"Measuring {node.name} ({plan.describe()})")
            return False
        choice = legal[0] if len(legal) == 1 else plan.candidate(plan.static_choice)
        if choice.name == routing.REGULAR_LOWERING:
            reasons = [c.reason for c in plan.candidates if c.reason]
            reason = plan.static_reason if len(legal) > 1 else "; ".join(reasons)
            logger.info(f"Skipping {plan.kind} replacement of {node.name}: {reason}")
            return False
        self._plans[node] = (plan, choice)
        only = "" if plan.regular_lowering_runs_it() else ", only implementation"
        logger.info(f"Replacing {node.name} ({choice.label}{only})")
        return True

    def _replace_node_with_triton(self, graph_module: GraphModule, node: Node) -> bool:
        """
        Replace an edge dialect node with a Triton kernel call.

        Args:
            graph_module: The graph module containing the node
            node: The node to replace

        Returns:
            Whether the node was replaced
        """
        plan, choice = self._plans.pop(node)
        return routing.replace(graph_module.graph, node, plan, choice) is not None
