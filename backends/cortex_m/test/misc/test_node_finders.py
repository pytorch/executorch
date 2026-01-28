# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.cortex_m.quantizer.node_finders as node_finders
import torch
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


def test_node_name_finder_multiple_names():
    """Test NodeNameNodeFinder with multiple names."""
    graph_module = _export_graph_module()

    node_finder = node_finders.NodeNameNodeFinder(["add", "sub"])
    nodes = list(node_finder.find_nodes(graph_module))
    node_names = {n.name for n in nodes}
    assert node_names == {"add", "sub"}


def test_node_name_finder_missing_name():
    """Test NodeNameNodeFinder returns no nodes for a missing name."""
    graph_module = _export_graph_module()

    node_finder = node_finders.NodeNameNodeFinder("not_in_graph")
    nodes = list(node_finder.find_nodes(graph_module))
    assert nodes == []


def test_node_target_finder_single_target():
    """Test NodeTargetNodeFinder with a single target."""
    graph_module = _export_graph_module()

    node_finder = node_finders.NodeTargetNodeFinder(torch.ops.aten.add.Tensor)
    nodes = list(node_finder.find_nodes(graph_module))
    assert len(nodes) == 1
    assert nodes[0].target == torch.ops.aten.add.Tensor


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


def test_input_node_finder():
    """Test InputNodeFinder finds all placeholder nodes."""
    graph_module = _export_graph_module()

    node_finder = node_finders.InputNodeFinder()
    nodes = list(node_finder.find_nodes(graph_module))

    node_names = [node.name for node in nodes]
    assert node_names == ["p_linear_weight", "p_linear_bias", "x"]
    assert all(node.op == "placeholder" for node in nodes)


def test_output_node_finder():
    """Test OutputNodeFinder finds the output node."""
    graph_module = _export_graph_module()

    node_finder = node_finders.OutputNodeFinder()
    nodes = list(node_finder.find_nodes(graph_module))

    assert len(nodes) == 1
    assert nodes[0].op == "output"


def test_module_name_finder():
    """Test ModuleNameNodeFinder finds nodes by module name."""
    graph_module = _export_graph_module()

    node_finder = node_finders.ModuleNameNodeFinder("linear")
    nodes = list(node_finder.find_nodes(graph_module))

    assert len(nodes) == 1
    assert nodes[0].target == torch.ops.aten.linear.default


def test_module_name_finder_missing_name():
    """Test ModuleNameNodeFinder returns no nodes for a missing name."""
    graph_module = _export_graph_module()

    node_finder = node_finders.ModuleNameNodeFinder("not_in_graph")
    nodes = list(node_finder.find_nodes(graph_module))
    assert nodes == []


def test_module_type_finder():
    """Test ModuleTypeNodeFinder finds nodes by module type."""
    graph_module = _export_graph_module()

    node_finder = node_finders.ModuleTypeNodeFinder(torch.nn.Linear)
    nodes = list(node_finder.find_nodes(graph_module))

    assert len(nodes) == 1
    assert nodes[0].target == torch.ops.aten.linear.default


def test_module_type_finder_missing_type():
    """Test ModuleTypeNodeFinder returns no nodes for a missing type."""
    graph_module = _export_graph_module()

    node_finder = node_finders.ModuleTypeNodeFinder(torch.nn.Conv2d)
    nodes = list(node_finder.find_nodes(graph_module))
    assert nodes == []
