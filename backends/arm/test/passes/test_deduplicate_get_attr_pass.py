# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm._passes.deduplicate_get_attr_pass import (
    DeduplicateGetAttrPass,
)
from torch.fx import Graph, GraphModule


def test_deduplicate_get_attr_splits_shared_node_users() -> None:
    root = torch.nn.Module()
    shared = torch.ones(2, 2)
    root.register_buffer("shared", shared)

    graph = Graph()
    x = graph.placeholder("x")
    attr = graph.get_attr("shared")
    first = graph.call_function(torch.ops.aten.add.Tensor, (x, attr))
    second = graph.call_function(torch.ops.aten.sub.Tensor, (first, attr))
    graph.output(second)
    graph_module = GraphModule(root, graph)

    result = DeduplicateGetAttrPass()(graph_module)

    assert result is not None
    assert result.modified

    get_attrs = list(graph_module.graph.find_nodes(op="get_attr"))
    assert len(get_attrs) == 2
    assert len({node.target for node in get_attrs}) == 2
    assert first.args[1] is get_attrs[0]
    assert second.args[1] is get_attrs[1]
    assert getattr(graph_module, get_attrs[0].target) is shared
    assert getattr(graph_module, get_attrs[1].target) is shared
