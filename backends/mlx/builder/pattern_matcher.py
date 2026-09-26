#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from typing import List, TYPE_CHECKING

from executorch.backends.mlx._logging import logger
from executorch.backends.mlx.builder.op_registry import PatternHandler

if TYPE_CHECKING:
    from executorch.backends.mlx.builder.op_registry import MLXOpRegistry
    from torch.export import ExportedProgram


class PatternMatcher:
    """
    Discovers and applies pattern handlers to an FX graph.

    Pattern handlers match multi-node subgraphs and lower them to optimized
    MLX operations. This class orchestrates the pattern discovery process:

    1. Iterates through all registered pattern types
    2. For each pattern, tries to match it against every node in the graph
    3. When a match is found, assigns handlers to the head and body nodes

    The ordering matters: patterns are matched before dead code elimination
    because some pattern body nodes (e.g., update_cache) have no users
    since they mutate in-place, but they're not dead.
    """

    def __init__(self, ep: ExportedProgram, registry: "MLXOpRegistry"):
        self.ep = ep
        self.registry = registry
        self._matches: List[PatternHandler] = []

    def find_patterns(self) -> List[PatternHandler]:
        """
        Find all pattern matches in the graph.

        Returns a list of PatternHandler instances, one for each match found.
        Patterns are tried in registration order.
        """
        self._matches = []
        for name in self.registry.patterns():
            self._find_pattern(name)
        return self._matches

    def _find_pattern(self, name: str) -> None:
        """Try to match a single pattern type against all nodes."""
        pattern_cls = self.registry.get_pattern_cls(name)
        if pattern_cls is None:
            return

        for n in self.ep.graph.nodes:
            handler = pattern_cls.maybe_create(self.ep, n)
            if handler is not None:
                logger.debug(f"Pattern {name} matched at node {n.name}")
                self._matches.append(handler)
