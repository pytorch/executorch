# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.cortex_m.quantizer.node_finders as node_finders
import pytest
import torch
from executorch.backends.transforms.decompose_sdpa import (
    DecomposeScaledDotProductAttention,
)
from torch.export import export


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor):
        x = x + x
        x = x - x
        x = self.linear(x)
        return x * 2


def _export_graph_module():
    mod = TestModule()
    inputs = (torch.ones(4),)
    return export(mod, inputs).graph_module


def test_node_name_finder_single_name():
    """Test NodeNameNodeFinder with a single name."""
    graph_module = _export_graph_module()

    node_finder = node_finders.NodeNameNodeFinder("add")
    nodes = list(node_finder.find_nodes(graph_module))
    assert len(nodes) == 1
    assert nodes[0].name == "add"
    assert str(node_finder) == "NodeNameNodeFinder targeting names: add"


def test_node_name_finder_multiple_names():
    """Test NodeNameNodeFinder with multiple names."""
    graph_module = _export_graph_module()

    node_finder = node_finders.NodeNameNodeFinder(["add", "sub"])
    nodes = list(node_finder.find_nodes(graph_module))
    node_names = {n.name for n in nodes}
    assert node_names == {"add", "sub"}
    assert str(node_finder) == "NodeNameNodeFinder targeting names: add, sub"


def test_node_name_finder_missing_name():
    """Test NodeNameNodeFinder returns no nodes for a missing name."""
    graph_module = _export_graph_module()

    node_finder = node_finders.NodeNameNodeFinder("not_in_graph")
    nodes = list(node_finder.find_nodes(graph_module))
    assert nodes == []
    assert str(node_finder) == "NodeNameNodeFinder targeting names: not_in_graph"


def test_node_target_finder_single_target():
    """Test NodeTargetNodeFinder with a single target."""
    graph_module = _export_graph_module()

    node_finder = node_finders.NodeTargetNodeFinder(torch.ops.aten.add.Tensor)
    nodes = list(node_finder.find_nodes(graph_module))
    assert len(nodes) == 1
    assert nodes[0].target == torch.ops.aten.add.Tensor
    assert (
        str(node_finder)
        == "NodeTargetNodeFinder targeting node targets: aten::add.Tensor"
    )


def test_node_target_finder_multiple_targets():
    """Test NodeTargetNodeFinder with multiple targets."""
    graph_module = _export_graph_module()

    node_finder = node_finders.NodeTargetNodeFinder(
        [torch.ops.aten.sub.Tensor, torch.ops.aten.add.Tensor]
    )
    nodes = list(node_finder.find_nodes(graph_module))
    targets = {n.target for n in nodes}
    assert targets == {torch.ops.aten.add.Tensor, torch.ops.aten.sub.Tensor}


def test_node_target_finder_missing_target():
    """Test NodeTargetNodeFinder returns no nodes for a missing target."""
    graph_module = _export_graph_module()

    node_finder = node_finders.NodeTargetNodeFinder(torch.ops.aten.relu.default)
    nodes = list(node_finder.find_nodes(graph_module))
    assert nodes == []
    assert str(node_finder) == "NodeTargetNodeFinder targeting node targets: aten::relu"


def test_node_target_activation_finder():
    """Test NodeTargetActivationNodeFinder skips activations not consuming the target."""

    class LinearRelu(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, x: torch.Tensor):
            return torch.relu(self.linear(torch.relu(x)))

    graph_module = export(LinearRelu(), (torch.ones(4),)).graph_module

    node_finder = node_finders.NodeTargetActivationNodeFinder(
        torch.ops.aten.linear.default, torch.ops.aten.relu.default
    )
    nodes = list(node_finder.find_nodes(graph_module))
    assert [n.name for n in nodes] == ["linear", "relu_1"]
    assert (
        str(node_finder)
        == "NodeTargetActivationNodeFinder targeting node target: aten::linear "
        "with activations: aten::relu"
    )


def test_node_target_activation_finder_skips_forked_target():
    """Test NodeTargetActivationNodeFinder skips activations of a target with other users."""

    class ForkedLinearRelu(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, x: torch.Tensor):
            y = self.linear(x)
            return torch.relu(y) + y

    graph_module = export(ForkedLinearRelu(), (torch.ones(4),)).graph_module

    node_finder = node_finders.NodeTargetActivationNodeFinder(
        torch.ops.aten.linear.default, torch.ops.aten.relu.default
    )
    nodes = list(node_finder.find_nodes(graph_module))
    assert [n.name for n in nodes] == ["linear"]


def test_global_node_finder():
    """Test GlobalNodeFinder finds all nodes."""
    graph_module = _export_graph_module()

    node_finder = node_finders.GlobalNodeFinder()
    nodes = list(node_finder.find_nodes(graph_module))

    node_names = [node.name for node in nodes]
    assert node_names == [
        "p_linear_weight",
        "p_linear_bias",
        "x",
        "add",
        "sub",
        "linear",
        "mul",
        "output",
    ]
    assert str(node_finder) == "GlobalNodeFinder targeting all nodes"


def test_input_node_finder():
    """Test InputNodeFinder finds all placeholder nodes."""
    graph_module = _export_graph_module()

    node_finder = node_finders.InputNodeFinder()
    nodes = list(node_finder.find_nodes(graph_module))

    node_names = [node.name for node in nodes]
    assert node_names == ["p_linear_weight", "p_linear_bias", "x"]
    assert all(node.op == "placeholder" for node in nodes)
    assert str(node_finder) == "InputNodeFinder targeting all placeholder nodes"


def test_output_node_finder():
    """Test OutputNodeFinder finds the output node."""
    graph_module = _export_graph_module()

    node_finder = node_finders.OutputNodeFinder()
    nodes = list(node_finder.find_nodes(graph_module))

    assert len(nodes) == 1
    assert nodes[0].op == "output"
    assert str(node_finder) == "OutputNodeFinder targeting the output node"


def test_module_name_finder():
    """Test ModuleNameNodeFinder finds nodes by module name."""
    graph_module = _export_graph_module()

    node_finder = node_finders.ModuleNameNodeFinder("linear")
    nodes = list(node_finder.find_nodes(graph_module))

    assert len(nodes) == 1
    assert nodes[0].target == torch.ops.aten.linear.default
    assert str(node_finder) == "ModuleNameNodeFinder targeting module names: linear"


def test_module_name_finder_missing_name():
    """Test ModuleNameNodeFinder returns no nodes for a missing name."""
    graph_module = _export_graph_module()

    node_finder = node_finders.ModuleNameNodeFinder("not_in_graph")
    nodes = list(node_finder.find_nodes(graph_module))
    assert nodes == []
    assert (
        str(node_finder) == "ModuleNameNodeFinder targeting module names: not_in_graph"
    )


def test_module_type_finder():
    """Test ModuleTypeNodeFinder finds nodes by module type."""
    graph_module = _export_graph_module()

    node_finder = node_finders.ModuleTypeNodeFinder(torch.nn.Linear)
    nodes = list(node_finder.find_nodes(graph_module))

    assert len(nodes) == 1
    assert nodes[0].target == torch.ops.aten.linear.default
    assert str(node_finder) == "ModuleTypeNodeFinder targeting module types: Linear"


def test_module_type_finder_missing_type():
    """Test ModuleTypeNodeFinder returns no nodes for a missing type."""
    graph_module = _export_graph_module()

    node_finder = node_finders.ModuleTypeNodeFinder(torch.nn.Conv2d)
    nodes = list(node_finder.find_nodes(graph_module))
    assert nodes == []
    assert str(node_finder) == "ModuleTypeNodeFinder targeting module types: Conv2d"


class AttentionAndUnrelatedMatmul(torch.nn.Module):
    def forward(self, q, k, v):
        first = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        second = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return first, second, torch.matmul(q, k.transpose(-2, -1))


def _attention_graph():
    inputs = tuple(torch.randn(1, 2, 4, 8) for _ in range(3))
    return export(AttentionAndUnrelatedMatmul(), inputs).module()


def test_sdpa_finders_match_native_calls_without_arm_metadata():
    graph_module = _attention_graph()
    found = list(node_finders.SDPAFinder().find_nodes(graph_module))
    assert len(found) == 2
    assert all(
        node.target == torch.ops.aten.scaled_dot_product_attention.default
        for node in found
    )
    for role in (None, "qk", "av"):
        assert not list(node_finders.SDPAMatmulFinder(role).find_nodes(graph_module))


def test_sdpa_finders_use_standard_provenance_and_dependencies():
    graph_module = DecomposeScaledDotProductAttention()(_attention_graph()).graph_module
    for index, node in enumerate(graph_module.graph.nodes):
        node.name = f"opaque_{index}"
    metadata = {node: dict(node.meta) for node in graph_module.graph.nodes}
    attention_nodes = set(node_finders.SDPAFinder().find_nodes(graph_module))
    products = set(node_finders.SDPAMatmulFinder().find_nodes(graph_module))
    scores = set(node_finders.SDPAMatmulFinder("qk").find_nodes(graph_module))
    values = set(node_finders.SDPAMatmulFinder("av").find_nodes(graph_module))
    assert len(products) == 4
    assert len(scores) == len(values) == 2
    assert not scores & values
    assert products == scores | values
    assert products <= attention_nodes
    unrelated = {
        node
        for node in graph_module.graph.nodes
        if node.target == torch.ops.aten.matmul.default
    }
    assert len(unrelated) == 1
    assert not unrelated & attention_nodes
    for node in graph_module.graph.nodes:
        assert node.meta == metadata[node]
        assert not any(key.startswith("_arm_sdpa") for key in node.meta)


def test_sdpa_finders_do_not_guess_missing_provenance():
    graph_module = DecomposeScaledDotProductAttention()(_attention_graph()).graph_module
    for node in graph_module.graph.nodes:
        node.meta.pop("source_fn_stack", None)
    assert not list(node_finders.SDPAFinder().find_nodes(graph_module))
    assert not list(node_finders.SDPAMatmulFinder("qk").find_nodes(graph_module))


def test_sdpa_matmul_finder_rejects_ambiguous_decomposition():
    graph_module = DecomposeScaledDotProductAttention()(_attention_graph()).graph_module
    score = next(node_finders.SDPAMatmulFinder("qk").find_nodes(graph_module))
    with graph_module.graph.inserting_after(score):
        extra = graph_module.graph.call_function(score.target, score.args)
        extra.meta = dict(score.meta)
    with pytest.raises(ValueError, match="Cannot identify QK and AV"):
        list(node_finders.SDPAMatmulFinder("qk").find_nodes(graph_module))


def test_sdpa_matmul_finder_rejects_unknown_role():
    with pytest.raises(ValueError, match="role must be"):
        node_finders.SDPAMatmulFinder("projection")


def test_sdpa_score_finder_keeps_operand_and_accumulator_policy_separate():
    graph_module = DecomposeScaledDotProductAttention()(_attention_graph()).graph_module
    scores = set(node_finders.SDPAScoreFinder().find_nodes(graph_module))
    matmuls = set(node_finders.SDPAMatmulFinder().find_nodes(graph_module))
    assert scores
    assert not scores & matmuls
    assert {node.target for node in scores} >= {torch.ops.aten._safe_softmax.default}
    for matmul in node_finders.SDPAMatmulFinder("qk").find_nodes(graph_module):
        assert not scores & set(matmul.all_input_nodes)
