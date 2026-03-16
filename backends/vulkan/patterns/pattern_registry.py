# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List, Optional

import torch

from executorch.exir import ExportedProgram

from torch.fx.passes.utils.matcher_utils import InternalMatch

GetGraphFn = Callable[[], List[torch.fx.GraphModule]]


class PatternMatch:
    __slots__ = ("input_nodes", "output_nodes", "all_nodes", "anchor_node")
    """
    The design of this class is based on InternalMatch from
    torch.fx.passes.utils.matcher_utils. It represents nodes in a graph that
    match a particular pattern.

    The reason to not use InternalMatch directly is to enable more (i.e. custom)
    methods to detect and represent matches other than through SubgraphMatcher.
    """

    def __init__(
        self,
        input_nodes: List[torch.fx.Node],
        output_nodes: List[torch.fx.Node],
        all_nodes: List[torch.fx.Node],
        anchor_node: Optional[torch.fx.Node] = None,
    ):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.all_nodes = all_nodes
        self.anchor_node = anchor_node


def create_pattern_match_from_internal_match(
    internal_match: InternalMatch,
) -> PatternMatch:
    return PatternMatch(
        internal_match.placeholder_nodes,
        internal_match.returning_nodes,
        list(internal_match.nodes_map.values()),
    )


CreateReplacementFn = Callable[
    [ExportedProgram, torch.fx.GraphModule, PatternMatch], None
]


DetectorFn = Callable[[torch.fx.Node], Optional[PatternMatch]]


class PatternEntry:
    def __init__(
        self,
        get_graphs_fn: Optional[GetGraphFn] = None,
        detector_fn: Optional[DetectorFn] = None,
        create_replacement_fn: Optional[CreateReplacementFn] = None,
    ):
        self.get_graphs_fn = get_graphs_fn
        self.detector_fn = detector_fn
        self.create_replacement_fn = create_replacement_fn

    def is_valid(self):
        return (
            self.get_graphs_fn is not None or self.detector_fn is not None
        ) and self.create_replacement_fn is not None


fusable_patterns: Dict[str, PatternEntry] = {}


def register_pattern_graph(pattern_name: str):
    def decorator(fn: GetGraphFn):
        if pattern_name not in fusable_patterns:
            fusable_patterns[pattern_name] = PatternEntry()

        # Cannot define both get_graphs_fn and detector_fn
        assert fusable_patterns[pattern_name].detector_fn is None
        fusable_patterns[pattern_name].get_graphs_fn = fn

        return fn

    return decorator


def register_pattern_detector(pattern_name: str):
    def decorator(fn: DetectorFn):
        if pattern_name not in fusable_patterns:
            fusable_patterns[pattern_name] = PatternEntry()

        # Cannot define both get_graphs_fn and detector_fn
        assert fusable_patterns[pattern_name].get_graphs_fn is None
        fusable_patterns[pattern_name].detector_fn = fn

        return fn

    return decorator


def register_pattern_replacement(pattern_name: str):
    def decorator(fn: CreateReplacementFn):
        if pattern_name not in fusable_patterns:
            fusable_patterns[pattern_name] = PatternEntry()

        fusable_patterns[pattern_name].create_replacement_fn = fn
        return fn

    return decorator
