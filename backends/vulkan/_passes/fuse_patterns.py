# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import Callable, List, Optional

import executorch.backends.vulkan.patterns as vk_patterns

import torch

from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

from torch.fx.passes.utils.matcher_utils import InternalMatch, SubgraphMatcher


def fuse_pattern(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    patterns: List[torch.fx.GraphModule],
    create_replacement_func: Callable,
) -> int:
    total_replaced = 0

    for pattern in patterns:
        sm = SubgraphMatcher(pattern.graph, ignore_literals=True)
        matches = list(sm.match(graph_module.graph))

        for partition_to_replace in matches:
            create_replacement_func(ep, graph_module, partition_to_replace)
            total_replaced += 1
            # Remove dead code so they won't be matched again
            graph_module.graph.eliminate_dead_code()

    return total_replaced


##
## Rotary Embedding
##


def identify_rotary_emb_io_nodes(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: InternalMatch,
) -> Optional[List[torch.fx.Node]]:
    # Get the input placeholders (xq, xk, freqs_cos, freqs_sin)
    placeholder_nodes = match.placeholder_nodes
    if len(placeholder_nodes) != 4:
        return None

    xq, xk, freqs_cos, freqs_sin = placeholder_nodes

    output_nodes = match.returning_nodes
    if len(output_nodes) != 2:
        return None

    xq_out, xk_out = output_nodes

    return [xq, xk, freqs_cos, freqs_sin, xq_out, xk_out]


def create_rotary_emb_custom_op(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: InternalMatch,
):
    io_nodes = identify_rotary_emb_io_nodes(ep, graph_module, match)
    if io_nodes is None:
        return

    assert len(io_nodes) == 6
    xq, xk, freqs_cos, freqs_sin, xq_out, xk_out = io_nodes

    # Create the custom op node
    with graph_module.graph.inserting_before(xq_out):
        rotary_emb_node = graph_module.graph.create_node(
            "call_function",
            exir_ops.edge.et_vk.apply_rotary_emb.default,
            args=(xq, xk, freqs_cos, freqs_sin),
        )

    # The custom op returns a tuple (xq_out, xk_out)
    # We need to extract the individual outputs
    with graph_module.graph.inserting_after(rotary_emb_node):
        getitem_0 = graph_module.graph.create_node(
            "call_function",
            operator.getitem,
            args=(rotary_emb_node, 0),
        )
        getitem_1 = graph_module.graph.create_node(
            "call_function",
            operator.getitem,
            args=(rotary_emb_node, 1),
        )

    if hasattr(xq_out, "meta") and "val" in xq_out.meta:
        getitem_0.meta["val"] = xq_out.meta["val"]
    if hasattr(xk_out, "meta") and "val" in xk_out.meta:
        getitem_1.meta["val"] = xk_out.meta["val"]

    xq_out.replace_all_uses_with(getitem_0)
    xk_out.replace_all_uses_with(getitem_1)


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
