# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.cortex_m.quantizer.node_finders as node_finders
import torch
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from executorch.backends.cortex_m.quantizer.pattern_checkers import PatternCheck
from executorch.backends.cortex_m.quantizer.pattern_matcher import PatternMatcher
from torch.export import export


class _TwoOpModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.relu(x + x)


def _export_two_op_graph_module():
    return export(_TwoOpModule(), (torch.ones(2, 2),)).graph_module


class _FullModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.full(x.shape, 1e20, dtype=x.dtype, device=x.device)


def _export_full_graph_module():
    return export(_FullModule(), (torch.ones(2, 2),)).graph_module


class _AlwaysPassCheck(PatternCheck):
    @classmethod
    def check_pattern(cls, pattern):
        return True

    @classmethod
    def check_quantization_config(cls, pattern, quantization_config):
        return True


class _AlwaysFailCheck(PatternCheck):
    @classmethod
    def check_pattern(cls, pattern):
        return False

    @classmethod
    def check_quantization_config(cls, pattern, quantization_config):
        return False


def _node_iter_for_targets(graph_module, targets):
    return node_finders.NodeTargetNodeFinder(targets).find_nodes(graph_module)


def _node_iter(graph_module):
    return _node_iter_for_targets(
        graph_module, [torch.ops.aten.add.Tensor, torch.ops.aten.relu.default]
    )


def _dummy_qconfig():
    return QuantizationConfig(None, None, None, None)


def test_matches_linear_chain_pattern():
    """Test basic pattern match functionality."""
    graph_module = _export_two_op_graph_module()
    support = {
        (torch.ops.aten.add.Tensor, torch.ops.aten.relu.default): _AlwaysPassCheck,
    }
    matcher = PatternMatcher(support)

    matches = list(
        matcher.find_pattern_matches(_node_iter(graph_module), _dummy_qconfig())
    )

    assert len(matches) == 1
    assert matches[0].accepted
    assert [n.target for n in matches[0].pattern] == [
        torch.ops.aten.add.Tensor,
        torch.ops.aten.relu.default,
    ]
    assert all(n.meta[PatternMatcher.Q_PATTERN_MATCHED_KEY] for n in matches[0].pattern)


def test_prefers_longest_available_pattern():
    """Test that when multiple patterns match, the longest pattern is preferred."""
    graph_module = _export_two_op_graph_module()
    support = {
        (torch.ops.aten.add.Tensor,): _AlwaysPassCheck,
        (torch.ops.aten.add.Tensor, torch.ops.aten.relu.default): _AlwaysPassCheck,
    }
    matcher = PatternMatcher(support)

    matches = list(
        matcher.find_pattern_matches(_node_iter(graph_module), _dummy_qconfig())
    )

    assert len(matches) == 1
    assert matches[0].accepted
    assert len(matches[0].pattern) == 2
    assert [n.target for n in matches[0].pattern] == [
        torch.ops.aten.add.Tensor,
        torch.ops.aten.relu.default,
    ]


def test_pattern_checker_can_reject_match():
    """Test basic pattern rejection capability"""
    graph_module = _export_two_op_graph_module()
    support = {
        (torch.ops.aten.add.Tensor, torch.ops.aten.relu.default): _AlwaysFailCheck,
    }
    matcher = PatternMatcher(support)

    matches = list(
        matcher.find_pattern_matches(_node_iter(graph_module), _dummy_qconfig())
    )

    assert len(matches) == 1
    assert not matches[0].accepted
    assert matches[0].message == PatternMatcher.REJECT_UNSUPPORTED_PATTERN


def test_rejects_longer_match_then_accepts_shorter_match():
    """Test that a shorter match is accepted if a longer match is rejected by the pattern checker and both are reported."""
    graph_module = _export_two_op_graph_module()
    support = {
        (torch.ops.aten.add.Tensor, torch.ops.aten.relu.default): _AlwaysFailCheck,
        (torch.ops.aten.add.Tensor,): _AlwaysPassCheck,
    }
    matcher = PatternMatcher(support)

    matches = list(
        matcher.find_pattern_matches(_node_iter(graph_module), _dummy_qconfig())
    )

    assert len(matches) == 2
    assert not matches[0].accepted
    assert matches[0].message == PatternMatcher.REJECT_UNSUPPORTED_PATTERN
    assert matches[1].accepted
    assert [n.target for n in matches[1].pattern] == [torch.ops.aten.add.Tensor]


def _get_node_by_target(graph_module, target):
    return next(n for n in graph_module.graph.nodes if n.target == target)


def _get_output_node(graph_module):
    return next(n for n in graph_module.graph.nodes if n.op == "output")


def test_missing_second_node_matches_first_node_pattern():
    """Test that a pattern going outside the selected nodes is non matched."""
    graph_module = _export_two_op_graph_module()
    support = {
        (torch.ops.aten.add.Tensor,): _AlwaysPassCheck,
        (torch.ops.aten.add.Tensor, torch.ops.aten.relu.default): _AlwaysPassCheck,
    }
    matcher = PatternMatcher(support)

    add_node = _get_node_by_target(graph_module, torch.ops.aten.add.Tensor)

    matches = list(matcher.find_pattern_matches(iter([add_node]), _dummy_qconfig()))

    assert len(matches) == 1
    assert matches[0].accepted
    assert [n.target for n in matches[0].pattern] == [torch.ops.aten.add.Tensor]
    assert add_node.meta[PatternMatcher.Q_PATTERN_MATCHED_KEY]


def test_missing_second_node_with_below_node_matches_first_node_pattern():
    """Similar to test_missing_second_node_matches_first_node_pattern but with an additional node below the matched node to ensure that the presence of additional nodes does not interfere with matching."""
    graph_module = _export_two_op_graph_module()
    support = {
        (torch.ops.aten.add.Tensor,): _AlwaysPassCheck,
        (torch.ops.aten.add.Tensor, torch.ops.aten.relu.default): _AlwaysPassCheck,
    }
    matcher = PatternMatcher(support)

    add_node = _get_node_by_target(graph_module, torch.ops.aten.add.Tensor)
    relu_node = _get_node_by_target(graph_module, torch.ops.aten.relu.default)
    output_node = _get_output_node(graph_module)

    matches = list(
        matcher.find_pattern_matches(iter([add_node, output_node]), _dummy_qconfig())
    )

    assert len(matches) == 2
    assert matches[0].accepted
    assert [n.target for n in matches[0].pattern] == [torch.ops.aten.add.Tensor]
    assert add_node.meta[PatternMatcher.Q_PATTERN_MATCHED_KEY]
    assert not relu_node.meta.get(PatternMatcher.Q_PATTERN_MATCHED_KEY, False)
    assert matches[1].accepted
    assert matches[1].pattern == [output_node]
    assert output_node.meta[PatternMatcher.Q_PATTERN_MATCHED_KEY]


def test_rejects_large_scalar_match():
    """Tests that patterns with large scalar constants are rejected regardless of pattern checker."""
    graph_module = _export_full_graph_module()
    support = {
        (torch.ops.aten.full.default,): _AlwaysPassCheck,
    }
    matcher = PatternMatcher(support)

    matches = list(
        matcher.find_pattern_matches(
            _node_iter_for_targets(graph_module, [torch.ops.aten.full.default]),
            _dummy_qconfig(),
        )
    )

    assert len(matches) == 1
    assert not matches[0].accepted
    assert matches[0].message == PatternMatcher.REJECT_LARGE_SCALAR


def test_accept_none_nodechecker():
    """Tests that patterns with None as the pattern checker are accepted."""
    graph_module = _export_two_op_graph_module()
    support = {
        (torch.ops.aten.add.Tensor, torch.ops.aten.relu.default): None,
    }
    matcher = PatternMatcher(support)

    matches = list(
        matcher.find_pattern_matches(_node_iter(graph_module), _dummy_qconfig())
    )

    assert len(matches) == 1
    assert matches[0].accepted


def test_reject_reported_once():
    """Tests that the pattern matcher reports a rejected pattern only once."""
    graph_module = _export_two_op_graph_module()
    support = {
        (torch.ops.aten.add.Tensor,): _AlwaysFailCheck,
        (
            torch.ops.aten.add.Tensor,
            torch.ops.aten.relu.default,
            torch.ops.aten.mul.Tensor,
        ): _AlwaysFailCheck,
    }
    matcher = PatternMatcher(support)

    matches = list(
        matcher.find_pattern_matches(_node_iter(graph_module), _dummy_qconfig())
    )

    assert len(matches) == 1
    assert not matches[0].accepted
