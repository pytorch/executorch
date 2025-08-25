# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import executorch.backends.vulkan.patterns.quantized_linear  # noqa

import executorch.backends.vulkan.patterns.rope  # noqa

import torch

from executorch.backends.vulkan.patterns.pattern_registry import (
    CreateReplacementFn,
    fusable_patterns,
    GetGraphFn,
    register_pattern_graph,
    register_pattern_replacement,
)

from executorch.backends.vulkan.patterns.rope import RotaryEmbeddingPattern

from executorch.exir import ExportedProgram

from torch.fx.passes.utils.matcher_utils import InternalMatch, SubgraphMatcher


__all__ = [
    "GetGraphFn",
    "CreateReplacementFn",
    "RotaryEmbeddingPattern",
    "fusable_patterns",
    "register_pattern_graph",
    "register_pattern_replacement",
]


def all_fusable_graph_patterns() -> List[torch.fx.GraphModule]:
    all_patterns = []
    for entry in fusable_patterns.values():
        if entry.get_graphs_fn is not None:
            all_patterns.extend(entry.get_graphs_fn())

    return all_patterns


def get_all_fusable_subgraphs(
    graph_module: torch.fx.GraphModule,
) -> List[InternalMatch]:
    fusable_subgraphs = []

    fuse_patterns = all_fusable_graph_patterns()
    for pattern in fuse_patterns:
        sm = SubgraphMatcher(pattern.graph, ignore_literals=True)
        matches = list(sm.match(graph_module.graph))
        fusable_subgraphs.extend(matches)

    return fusable_subgraphs


def create_replacement_for_pattern(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    patterns: List[torch.fx.GraphModule],
    create_replacement_func: CreateReplacementFn,
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


def replace_all_fusable_subgraphs(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
) -> int:
    total_replaced = 0

    for entry in fusable_patterns.values():
        if entry.get_graphs_fn is not None and entry.create_replacement_fn is not None:
            total_replaced += create_replacement_for_pattern(
                ep,
                graph_module,
                entry.get_graphs_fn(),
                # pyre-ignore[6]
                entry.create_replacement_fn,
            )

    return total_replaced
