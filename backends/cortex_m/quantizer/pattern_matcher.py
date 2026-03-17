# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import Iterator, List, Optional

from executorch.backends.arm.quantizer.quantization_annotator import _is_large_scalar

from executorch.backends.cortex_m.quantizer.pattern_checkers import PatternCheck
from executorch.backends.cortex_m.quantizer.quantization_configs import (
    CortexMQuantizationConfig,
    QuantizationConfig,
)
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
        support_dict_name: An optional name for the support dict, used for logging.
    """

    Q_PATTERN_MATCHED_KEY = "quantizer_matched"
    REJECT_PREVIOUSLY_ANNOTATED = "Tried annotating already quantized node."
    REJECT_UNSUPPORTED_PATTERN = (
        "Tried annotating unsupported configuration of operators"
    )
    REJECT_UNSUPPORTED_QCONFIG = "Tried annotating unsupported quantization config"
    REJECT_LARGE_SCALAR = "Tried annotating a large constant scalar value that is not supported for quantization."

    def __init__(
        self,
        support_dict: dict[tuple[OpOverload, ...], PatternCheck],
        support_dict_name: str | None = None,
    ):
        self.support_dict = support_dict
        self.support_dict_name = support_dict_name

        self.max_pattern_len = max(
            (len(pattern) for pattern in support_dict.keys()), default=0
        )

    def _validate_match(
        self,
        match: List[Node],
        quantization_config: CortexMQuantizationConfig,
    ) -> Optional[PatternMatchResult]:
        """
        Returns a PatternMatchResult when the pattern structurally matches, with
        status indicating accept/reject. Returns None if there is no match.
        """

        # Reject match if it contains a node that has already been matched as part of another pattern.
        if any(node.meta.get(self.Q_PATTERN_MATCHED_KEY, False) for node in match):
            return PatternMatchResult(match, False, self.REJECT_PREVIOUSLY_ANNOTATED)

        # Reject match if it contains a node that has an input which is too large to be quantized
        if any(_is_large_scalar(node, node.graph.owning_module) for node in match):
            return PatternMatchResult(match, False, self.REJECT_LARGE_SCALAR)

        if all(node.op in ("placeholder", "output") for node in match):
            # Accept matches of length 1 that are just placeholders or outputs
            for node in match:
                node.meta[self.Q_PATTERN_MATCHED_KEY] = True
            return PatternMatchResult(match, True)

        key = tuple([n.target for n in match])
        pattern_checker = self.support_dict.get(key, None)

        if pattern_checker is not None:
            pattern_ok = pattern_checker.check_pattern(match)
            if not pattern_ok:
                return PatternMatchResult(match, False, self.REJECT_UNSUPPORTED_PATTERN)

            qconfig_ok = pattern_checker.check_quantization_config(
                match, quantization_config
            )
            if not qconfig_ok:
                return PatternMatchResult(match, False, self.REJECT_UNSUPPORTED_QCONFIG)

        for node in match:
            node.meta[self.Q_PATTERN_MATCHED_KEY] = True
        return PatternMatchResult(match, True)

    def _get_match(self, node_queue: List[Node]) -> List[Node]:
        """
        Returns the longest pattern match starting at the front of the queue.
        """
        if node_queue[0].op in ("placeholder", "output"):
            return [node_queue[0]]

        pattern_key = tuple(n.target for n in node_queue)
        while pattern_key:
            if pattern_key in self.support_dict:
                return node_queue[: len(pattern_key)]
            else:
                pattern_key = pattern_key[:-1]

        return []

    def _get_matches(
        self, node_queue: List[Node], quantization_config: QuantizationConfig
    ) -> List[PatternMatchResult]:
        """
        Returns the longest accepted match starting at the first node of the queue as well as longer rejected matches.
        """
        matches = []
        accepted = False
        max_match_length = len(node_queue)

        while max_match_length > 0 and not accepted:
            match = self._get_match(node_queue[:max_match_length])
            max_match_length = (
                len(match) - 1
            )  # Look for shorter matches in the next iter if no accepted match found

            if match:
                validated_match = self._validate_match(match, quantization_config)
                accepted = validated_match.accepted
                matches.append(validated_match)

        return matches

    def _dequeue_and_get_matches(
        self, node_queue: List[Node], quantization_config: QuantizationConfig
    ) -> List[PatternMatchResult]:
        """
        Dequeues the longest accepted match starting at the first node of the queue, and returns all potential matches that were checked (rejected ones). If no match is found, simply dequeues the first node and returns an empty list.
        """
        potential_matches = self._get_matches(node_queue, quantization_config)
        accepted_matches = [m for m in potential_matches if m.accepted]
        assert (
            len(accepted_matches) <= 1
        ), "_get_matches should only accept the longest possible match, but multiple accepted matches were found."

        if len(accepted_matches) == 0:
            node_queue.pop(0)
        else:
            del node_queue[: len(accepted_matches[0].pattern)]

        return potential_matches

    def find_pattern_matches(
        self, nodes: Iterator[Node], quantization_config: CortexMQuantizationConfig
    ) -> Iterator[PatternMatchResult]:
        """
        Match all given patterns in the graph and return match results with
        acceptance/rejection status.
        Each node can only be part of one match, larger patterns are prioritized.
        Currently only linear patterns (single chain) are supported.

        Q_PATTERN_MATCHED_KEY is set to True in node.meta to track which nodes have
        already been matched.
        """

        node = next(nodes, None)
        node_queue = []
        while node is not None:
            potential_matches = []
            node_queue.append(node)
            next_node = next(nodes, None)
            node_users = list(node.users)

            # If there is a fork or gap in the nodes iterator, empty the queue
            if (len(node_users) != 1) or (node_users[0] != next_node):
                while node_queue:
                    new_matches = self._dequeue_and_get_matches(
                        node_queue, quantization_config
                    )
                    potential_matches.extend(new_matches)

            # When que reach the max length, search for match starting at the front of the queue
            elif len(node_queue) >= self.max_pattern_len:
                potential_matches = self._dequeue_and_get_matches(
                    node_queue, quantization_config
                )

            # Report all pattern matches, also rejected ones for debugging purposes
            for match in potential_matches:
                yield match

            node = next_node
