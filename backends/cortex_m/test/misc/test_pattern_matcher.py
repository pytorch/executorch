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


class _AlwaysPassCheck(PatternCheck):
    @classmethod
    def check_pattern(cls, pattern):
        return True

    @classmethod
    def check_quantization_config(cls, quantization_config):
        return True


class _AlwaysFailCheck(PatternCheck):
    @classmethod
    def check_pattern(cls, pattern):
        return False

    @classmethod
    def check_quantization_config(cls, quantization_config):
        return False


def _node_iter(graph_module):
    return node_finders.NodeTargetNodeFinder(
        [torch.ops.aten.add.Tensor, torch.ops.aten.relu.default]
    ).find_nodes(graph_module)


def _dummy_qconfig():
    return QuantizationConfig(None, None, None, None)


def test_matches_linear_chain_pattern():
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


def test_filter_fn_blocks_match():
    graph_module = _export_two_op_graph_module()
    support = {(torch.ops.aten.add.Tensor,): _AlwaysPassCheck}
    matcher = PatternMatcher(
        support, filter_fn=lambda node: node.target == torch.ops.aten.add.Tensor
    )

    matches = list(
        matcher.find_pattern_matches(_node_iter(graph_module), _dummy_qconfig())
    )

    assert len(matches) == 1
    assert not matches[0].accepted
    assert matches[0].message == PatternMatcher.REJECT_FILTERED_OUT


def test_pattern_checker_can_reject_match():
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
