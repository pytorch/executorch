# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Collection, List

import executorch.backends.vulkan.patterns.conv2d_with_epilogue  # noqa

import executorch.backends.vulkan.patterns.quantized_binary  # noqa

import executorch.backends.vulkan.patterns.quantized_convolution  # noqa

import executorch.backends.vulkan.patterns.quantized_embedding  # noqa

import executorch.backends.vulkan.patterns.quantized_linear  # noqa

import executorch.backends.vulkan.patterns.quantized_pixel_shuffle  # noqa

import executorch.backends.vulkan.patterns.quantized_unary  # noqa

import executorch.backends.vulkan.patterns.rms_norm  # noqa

import executorch.backends.vulkan.patterns.rope  # noqa

import executorch.backends.vulkan.patterns.rope_hf  # noqa

import executorch.backends.vulkan.patterns.sdpa  # noqa

import executorch.backends.vulkan.patterns.select_as_symint  # noqa

import executorch.backends.vulkan.patterns.swiglu  # noqa

import torch

from executorch.backends.vulkan.patterns.pattern_registry import (
    create_pattern_match_from_internal_match,
    CreateReplacementFn,
    DetectorFn,
    fusable_patterns,
    GetGraphFn,
    PatternEntry,
    PatternMatch,
    register_pattern_detector,
    register_pattern_graph,
    register_pattern_replacement,
)

from executorch.backends.vulkan.patterns.rope import RotaryEmbeddingPattern
from executorch.backends.vulkan.patterns.rope_hf import HfRotaryEmbeddingPattern

from executorch.exir import ExportedProgram

from torch.fx.passes.utils.matcher_utils import SubgraphMatcher


__all__ = [
    "PatternMatch",
    "GetGraphFn",
    "DetectorFn",
    "CreateReplacementFn",
    "RotaryEmbeddingPattern",
    "HfRotaryEmbeddingPattern",
    "fusable_patterns",
    "register_pattern_graph",
    "register_pattern_detector",
    "register_pattern_replacement",
]


def enabled_patterns(skip_patterns: Collection[str] = ()) -> List[PatternEntry]:
    return [
        entry for name, entry in fusable_patterns.items() if name not in skip_patterns
    ]


def all_fusable_graph_patterns(
    skip_patterns: Collection[str] = (),
) -> List[torch.fx.GraphModule]:
    all_patterns = []
    for entry in enabled_patterns(skip_patterns):
        if entry.get_graphs_fn is not None:
            all_patterns.extend(entry.get_graphs_fn())

    return all_patterns


def get_all_fusable_subgraphs(
    graph_module: torch.fx.GraphModule,
    skip_patterns: Collection[str] = (),
) -> List[PatternMatch]:
    fusable_subgraphs = []

    fuse_patterns = all_fusable_graph_patterns(skip_patterns)
    for pattern in fuse_patterns:
        sm = SubgraphMatcher(pattern.graph, ignore_literals=True)
        matches = list(sm.match(graph_module.graph))
        for match in matches:
            fusable_subgraphs.append(create_pattern_match_from_internal_match(match))

    entries = enabled_patterns(skip_patterns)
    for node in graph_module.graph.nodes:
        for entry in entries:
            if entry.detector_fn is not None:
                maybe_match = entry.detector_fn(node)
                if maybe_match is not None:
                    fusable_subgraphs.append(maybe_match)

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
            pattern = create_pattern_match_from_internal_match(partition_to_replace)
            if create_replacement_func(ep, graph_module, pattern) is False:
                continue
            total_replaced += 1
            # Remove dead code so they won't be matched again
            graph_module.graph.eliminate_dead_code()

    return total_replaced


def replace_all_fusable_subgraphs(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    skip_patterns: Collection[str] = (),
) -> int:
    total_replaced = 0
    entries = enabled_patterns(skip_patterns)

    # Handle patterns identified with SubgraphMatcher
    for entry in entries:
        if entry.get_graphs_fn is not None and entry.create_replacement_fn is not None:
            total_replaced += create_replacement_for_pattern(
                ep,
                graph_module,
                entry.get_graphs_fn(),
                # pyre-ignore[6]
                entry.create_replacement_fn,
            )

    # Handle patterns identified with custom detector function
    for node in graph_module.graph.nodes:
        replaced_before = total_replaced
        for entry in entries:
            if (
                entry.detector_fn is not None
                and entry.create_replacement_fn is not None
            ):
                maybe_match = entry.detector_fn(node)
                if maybe_match is not None:
                    assert entry.create_replacement_fn is not None
                    if (
                        entry.create_replacement_fn(ep, graph_module, maybe_match)
                        is not False
                    ):
                        total_replaced += 1
        if total_replaced > replaced_before:
            # Remove dead code so they won't be matched again
            graph_module.graph.eliminate_dead_code()

    graph_module.graph.eliminate_dead_code()
    return total_replaced
