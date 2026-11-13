# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import io
import pickle
import unittest
from dataclasses import FrozenInstanceError

import torch
from executorch.exir._node_provenance import freeze_node_provenance
from executorch.exir.pass_base import ExportPass
from torch.fx.traceback import NodeSource, NodeSourceAction


class TestNodeProvenance(unittest.TestCase):
    def test_shared_history_is_immutable_and_isolated(self):
        graph = torch.fx.Graph()
        node = graph.placeholder("x")
        ancestor = NodeSource(node, "original", NodeSourceAction.CREATE)
        node.meta["from_node"] = [ancestor]
        source = NodeSource(node, "first_pass", NodeSourceAction.REPLACE)
        node.meta["from_node"] = [source, source]
        expected = source.to_dict()
        other_meta = {"shape": [2, 3]}
        node.meta["custom"] = other_meta

        freeze_node_provenance(graph)
        frozen = node.meta["from_node"][0]
        self.assertIs(frozen, node.meta["from_node"][1])
        self.assertEqual(frozen.to_dict(), expected)
        self.assertEqual(frozen, source)
        self.assertEqual(hash(frozen), hash(source))
        self.assertEqual(repr(frozen), repr(source))
        self.assertIs(node.meta["custom"], other_meta)

        copied = copy.deepcopy(node.meta["from_node"])
        self.assertIsNot(copied, node.meta["from_node"])
        self.assertIs(copied[0], frozen)
        next_source = NodeSource(node, "next_pass", NodeSourceAction.CREATE)
        self.assertIs(next_source.from_node[0], frozen)
        self.assertIsNot(next_source.from_node, node.meta["from_node"])
        copied.clear()
        self.assertEqual(len(node.meta["from_node"]), 2)
        source.pass_name = "changed"
        source.action.clear()
        source.from_node.clear()
        source.node_info.name = "changed"
        self.assertEqual(frozen.to_dict(), expected)
        with self.assertRaises(FrozenInstanceError):
            frozen.pass_name = "changed"
        with self.assertRaises(AttributeError):
            frozen.from_node.append(ancestor)
        with self.assertRaises(AttributeError):
            frozen.action.append(NodeSourceAction.REPLACE)
        with self.assertRaises(AttributeError):
            frozen.node_info.name = "changed"

        exported = frozen.to_dict()
        exported["from_node"].clear()
        self.assertEqual(frozen.to_dict(), expected)
        self.assertEqual(pickle.loads(pickle.dumps(frozen)).to_dict(), expected)
        reconstructed = NodeSource._from_dict(expected)
        self.assertEqual(reconstructed, frozen)
        freeze_node_provenance(graph)
        self.assertIs(node.meta["from_node"][0], frozen)

    def test_replay_preserves_ancestry_and_export_round_trip(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.sin(x) + x

        inputs = (torch.randn(2, 3),)
        ep = torch.export.export(Model(), inputs, strict=True)
        graph_module = ep.graph_module
        source_node = next(
            n for n in graph_module.graph.nodes if n.op == "call_function"
        )
        marker = NodeSource(source_node, "original", NodeSourceAction.CREATE)
        source_node.meta["from_node"] = [marker]
        expected = marker.to_dict()
        for index in range(4):
            source_node.meta["from_node"] = [
                NodeSource(source_node, f"pass_{index}", NodeSourceAction.CREATE)
            ]
        expected_history = source_node.meta["from_node"][0].to_dict()
        for _ in range(4):
            graph_module = ExportPass()(graph_module).graph_module

        output_node = next(
            n for n in graph_module.graph.nodes if n.op == "call_function"
        )
        history = output_node.meta["from_node"][0]
        self.assertEqual(history.to_dict(), expected_history)
        depth = 0
        while history.pass_name != "original":
            depth += 1
            history = history.from_node[0]
        self.assertGreaterEqual(depth, 4)
        self.assertEqual(history.to_dict(), expected)
        torch.testing.assert_close(graph_module(*inputs), ep.graph_module(*inputs))

        branch = copy.deepcopy(graph_module)
        branch_node = next(n for n in branch.graph.nodes if n.op == "call_function")
        self.assertIs(
            branch_node.meta["from_node"][0], output_node.meta["from_node"][0]
        )

        ep._graph_module = graph_module
        buffer = io.BytesIO()
        torch.export.save(ep, buffer)
        buffer.seek(0)
        loaded = torch.export.load(buffer)
        loaded_node = next(n for n in loaded.graph.nodes if n.op == "call_function")
        self.assertEqual(
            [s.to_dict() for s in loaded_node.meta["from_node"]],
            [s.to_dict() for s in output_node.meta["from_node"]],
        )
        torch.testing.assert_close(loaded.module()(*inputs), Model()(*inputs))


if __name__ == "__main__":
    unittest.main()
