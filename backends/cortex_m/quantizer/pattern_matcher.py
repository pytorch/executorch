# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict

from dataclasses import dataclass
from typing import Iterator, List, Optional

from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from executorch.backends.cortex_m.quantizer.pattern_checkers import PatternCheck
from torch._ops import OpOverload
from torch.fx import Node


@dataclass(frozen=True)
class PatternMatchResult:
    pattern: List[Node]
    accepted: bool
    message: Optional[str] = None


class PatternMatcher:
    """
    Find supported patterns in a sequence of nodes.

    Attributes:
        support_dict: A dictionary mapping patterns (tuples of operator overloads) to
                  PatternCheck instances that validate the patterns.
        filter_fn: A global filter applied over all nodes to exclude them from matching.
    """

    Q_PATTERN_MATCHED_KEY = "quantizer_matched"
    REJECT_PREVIOUSLY_ANNOTATED = "Tried annotating already quantized node."
    REJECT_FILTERED_OUT = "Node filtered out by global filter."
    REJECT_UNSUPPORTED_PATTERN = (
        "Tried annotating unsupported configuration of operators"
    )
    REJECT_UNSUPPORTED_QCONFIG = "Tried annotating unsupported quantization config"

    def __init__(
        self,
        support_dict: dict[tuple[OpOverload, ...], PatternCheck],
        filter_fn=lambda node: False,
        support_dict_name: str | None = None,
    ):
        self.support_dict = support_dict
        self.filter_fn = filter_fn
        self.support_dict_name = support_dict_name

        self.patterns_by_first = defaultdict(list)
        for p in sorted(support_dict.keys(), key=len, reverse=True):
            self.patterns_by_first[p[0]].append(p)

    def check_node(
        self, node: Optional[Node], target: OpOverload
    ) -> tuple[bool, Optional[str]]:
        """
        Return true if the node is a valid match for the given target.
        """
        if node is None:
            return False, None
        if not node.target == target:
            return False, None
        if node.meta.get(self.Q_PATTERN_MATCHED_KEY, False):
            return False, self.REJECT_PREVIOUSLY_ANNOTATED
        if self.filter_fn(node):
            return False, self.REJECT_FILTERED_OUT

        return True, None

    def check_pattern(
        self,
        node: Optional[Node],
        pattern: List[OpOverload],
        quantization_config: QuantizationConfig,
    ) -> Optional[PatternMatchResult]:
        """
        Returns a PatternMatchResult when the pattern structurally matches, with
        status indicating accept/reject. Returns None if there is no match.
        """
        match: List[Node] = []

        for pattern_target in pattern:
            node_ok, rejection_reason = self.check_node(node, pattern_target)
            if not node_ok:
                if rejection_reason is None:
                    return None
                return PatternMatchResult([node], False, rejection_reason)

            match.append(node)
            node = list(node.users)[0] if len(node.users) > 0 else None

        key = tuple([n.target for n in match])
        pattern_checker = self.support_dict.get(key, None)
        if pattern_checker:
            pattern_ok = pattern_checker.check_pattern(match)
            if not pattern_ok:
                return PatternMatchResult(match, False, self.REJECT_UNSUPPORTED_PATTERN)

            qconfig_ok = pattern_checker.check_quantization_config(quantization_config)
            if not qconfig_ok:
                return PatternMatchResult(match, False, self.REJECT_UNSUPPORTED_QCONFIG)

        return PatternMatchResult(match, True)

    def find_pattern_matches(
        self, nodes: Iterator[Node], quantization_config: QuantizationConfig
    ) -> Iterator[PatternMatchResult]:
        """
        Match all given patterns in the graph and return match results with
        acceptance/rejection status.
        Each node can only be part of one match, larger patterns are prioritized.
        Currently only linear patterns (single chain) are supported.

        Q_PATTERN_MATCHED_KEY is set to True in node.meta to track which nodes have
        already been matched.
        """

        for node in nodes:
            if node.meta.get(self.Q_PATTERN_MATCHED_KEY, False):
                yield PatternMatchResult(
                    [node], False, self.REJECT_PREVIOUSLY_ANNOTATED
                )  # Reject already matched nodes
                continue
            if node.op == "placeholder" or node.op == "output":
                node.meta[self.Q_PATTERN_MATCHED_KEY] = True
                yield PatternMatchResult(
                    [node], True
                )  # Always accept placeholders and outputs

            for pattern in self.patterns_by_first.get(node.target, []):
                match_or_none = self.check_pattern(node, pattern, quantization_config)
                if match_or_none is None:
                    continue  # No match, try next pattern
                if match_or_none.accepted:
                    for _ in range(len(match_or_none.pattern) - 1):
                        next(nodes)  # Fast-forward iterator to skip matched nodes
                    for matched_node in match_or_none.pattern:
                        matched_node.meta[self.Q_PATTERN_MATCHED_KEY] = True
                    yield match_or_none  # Accepted pattern found, break to skip checking remaining patterns
                    break
                else:
                    yield match_or_none  # Rejected pattern found, keep searching
