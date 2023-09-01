# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Callable, Dict, List, Optional, Union

import torch

from executorch.backends.qnnpack.partition.support_patterns import (
    get_dynamic_quant_addmm_with_view_copy_graph,
    get_dynamic_quant_addmm_without_view_copy_graph,
    get_dynamic_quant_mm_with_view_copy_graph,
    get_dynamic_quant_mm_without_view_copy_graph,
)
from executorch.backends.qnnpack.qnnpack_preprocess import QnnpackBackend
from executorch.backends.transforms.addmm_mm_to_linear import (
    apply_addmm_mm_to_linear_transform,
)
from executorch.exir.backend.partitioner import DelegationSpec, Partitioner
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class _BasePartitioner(Partitioner):
    """
    Graph based partitioner base for on QNNPACK backend.
    """

    def __init__(self, delegate_name, patterns):
        self.patterns = patterns

        self.delegation_spec = DelegationSpec(delegate_name, [])
        self.partition_tags: Dict[str, DelegationSpec] = {}

    @staticmethod
    def check_partitions(partitions: Union[dict, list]) -> None:
        """
        Warn users if there aren't any matches
        """
        pl = len(partitions)
        if pl == 0:
            log.warning("Nothing can be partitioned!")
        else:
            log.info(f"Found {pl} subgraphs to be partitioned.")

    def partition(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        raise NotImplementedError("This is not meant to be used directly.")
        return graph_module


class _SingleOpDelegatePartitioner(_BasePartitioner):
    """
    Graph based partitioner base for a single "op" or "node" or a pattern match for QNNPACK backend.
    This is tailored for DQLinear where QNNPACK delegates prefers to have a single DQLinear node in the graph.
    """

    def __init__(
        self,
        delegate_name,
        patterns,
        transforms: Optional[List[Callable[[torch.fx.Graph], torch.fx.Graph]]] = None,
    ):
        """
        @param transforms: Optional list of transforms that will be applied to the graph before running the partitioner.
        """
        super().__init__(delegate_name, patterns)
        self.transforms = transforms

    # override
    def partition(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        # TODO delete this since we are not allowed to do this
        if self.transforms is not None:
            for transform in self.transforms:  # pyre-ignore
                graph_module.graph = transform(graph_module.graph)

        matches = [
            match
            for matches in (
                SubgraphMatcher(pattern, ignore_literals=True).match(graph_module.graph)
                for pattern in self.patterns
            )
            for match in matches
        ]

        match_sets = [
            {
                node_in_graph
                for (node_in_pattern, node_in_graph) in match.nodes_map.items()
                if (
                    node_in_pattern.op != "placeholder"
                    and node_in_graph.op != "placeholder"
                )
            }
            for match in matches
        ]

        # Sort match sets in descending order of length so that any match sets
        # which are supersets of other match sets are processed first
        match_sets = sorted(match_sets, key=len, reverse=True)

        self.check_partitions(match_sets)

        # Mapping from delegation tag to match set
        tag_mapping = {}

        for (partition_id, match_set) in enumerate(match_sets):
            delegation_tag = f"tag{partition_id}"
            for node in match_set:
                if "delegation_tag" in node.meta:
                    # This node already has delegation tag assigned.
                    # Check that the current match set is a subset of the one
                    # used to assign its delegation tag, then skip this match
                    # set. We have this check to ensure there are no pairs of
                    # match sets where they are overlapping but neither is a
                    # subset of the other.
                    if not match_set.issubset(tag_mapping[node.meta["delegation_tag"]]):
                        raise AssertionError(
                            f"Found match sets which are overlapping but neither is a subset of the other: {match_set}, {tag_mapping[node.meta['delegation_tag']]}"
                        )
                    break
                node.meta["delegation_tag"] = delegation_tag
            self.partition_tags[delegation_tag] = self.delegation_spec
            tag_mapping[delegation_tag] = match_set

        return graph_module


class QnnpackPartitioner(_SingleOpDelegatePartitioner):
    def __init__(self) -> None:
        qnnp_patterns = [
            get_dynamic_quant_addmm_with_view_copy_graph(),
            get_dynamic_quant_addmm_without_view_copy_graph(),
            get_dynamic_quant_mm_with_view_copy_graph(),
            get_dynamic_quant_mm_without_view_copy_graph(),
            # Maybe there is a better way to handle dynamic shape
            # However, if we want to decouple partitioner from how the
            # graph was generated we need to capture all the ways in
            # which graph is generated _that_ can affect partitioner.
            get_dynamic_quant_addmm_with_view_copy_graph(dynamic_shape=True),
            get_dynamic_quant_addmm_without_view_copy_graph(dynamic_shape=True),
            get_dynamic_quant_mm_with_view_copy_graph(dynamic_shape=True),
            get_dynamic_quant_mm_without_view_copy_graph(dynamic_shape=True),
        ]
        super().__init__(
            QnnpackBackend.__name__, qnnp_patterns, [apply_addmm_mm_to_linear_transform]
        )
