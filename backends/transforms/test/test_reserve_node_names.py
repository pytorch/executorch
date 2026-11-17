# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.transforms.utils import reserve_node_names


class TestReserveNodeNames(unittest.TestCase):
    def test_new_node_does_not_take_an_existing_name(self):
        class Model(torch.nn.Module):
            def forward(self, x, mul):
                return x * 2 + mul

        # The input takes the name "mul", so export renames the multiply to
        # "mul_1" without recording that name in the graph's namespace.
        ep = torch.export.export(Model(), (torch.randn(2), torch.randn(2)))
        graph = ep.graph_module.graph
        self.assertIn("mul_1", [node.name for node in graph.nodes])
        # A skip, not a failure: a torch update that records the name is fine.
        if "mul_1" in graph._graph_namespace._used_names:
            self.skipTest(
                "export now records the names it renames, so this no longer "
                "tests reserve_node_names; see "
                "https://github.com/pytorch/pytorch/issues/202927"
            )

        reserve_node_names(ep.graph_module)
        (add,) = graph.find_nodes(op="call_function", target=torch.ops.aten.add.Tensor)
        with graph.inserting_after(add):
            node = graph.create_node(
                "call_function", torch.ops.aten.neg.default, (add,), name="mul_1"
            )
        self.assertNotEqual(node.name, "mul_1")
        graph.lint()
