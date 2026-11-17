# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import io
import pickle
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import torch
from executorch.exir._node_provenance import (
    _ImmutableNodeSource,
    freeze_node_provenance,
)
from executorch.exir.pass_base import ExportPass
from torch._higher_order_ops.map import map as torch_map
from torch.fx.traceback import NodeSource, NodeSourceAction


class Model(torch.nn.Module):
    def forward(self, inputs):
        return torch.sin(inputs) + inputs


class TestNodeProvenance(unittest.TestCase):
    def setUp(self):
        self.graph = torch.fx.Graph()
        self.node = self.graph.placeholder("inputs")
        self.ancestor = NodeSource(self.node, "original", NodeSourceAction.CREATE)
        self.node.meta["from_node"] = [self.ancestor]
        self.source = NodeSource(self.node, "first_pass", NodeSourceAction.REPLACE)
        self.node.meta["from_node"] = [self.source, self.source]

    def test_shared_history_deepcopy_does_not_traverse_ancestors(self):
        freeze_node_provenance(self.graph)
        frozen = self.node.meta["from_node"][0]
        self.assertIs(frozen, self.node.meta["from_node"][1])
        with patch.object(
            _ImmutableNodeSource,
            "__reduce__",
            side_effect=AssertionError("deepcopy traversed frozen history"),
        ):
            copied = copy.deepcopy(self.node.meta["from_node"])
            next_source = NodeSource(self.node, "next_pass", NodeSourceAction.CREATE)
        self.assertIsNot(copied, self.node.meta["from_node"])
        self.assertIs(copied[0], frozen)
        self.assertIs(next_source.from_node[0], frozen)
        self.assertIsNot(next_source.from_node, self.node.meta["from_node"])
        copied.clear()
        self.assertEqual(len(self.node.meta["from_node"]), 2)

    def test_snapshots_are_isolated_from_original_metadata(self):
        expected = self.source.to_dict()
        other_meta = {"shape": [2, 3]}
        self.node.meta["custom"] = other_meta
        freeze_node_provenance(self.graph)
        frozen = self.node.meta["from_node"][0]
        self.source.pass_name = "changed"
        self.source.action.clear()
        self.source.from_node.clear()
        self.source.node_info.name = "changed"
        self.assertEqual(frozen.to_dict(), expected)
        self.assertIs(self.node.meta["custom"], other_meta)

    def test_snapshots_are_immutable(self):
        freeze_node_provenance(self.graph)
        frozen = self.node.meta["from_node"][0]
        with self.assertRaises(FrozenInstanceError):
            frozen.pass_name = "changed"
        with self.assertRaises(AttributeError):
            frozen.from_node.append(self.ancestor)
        with self.assertRaises(AttributeError):
            frozen.action.append(NodeSourceAction.REPLACE)
        with self.assertRaises(AttributeError):
            frozen.node_info.name = "changed"

    def test_cached_serialization_is_isolated(self):
        expected = self.source.to_dict()
        freeze_node_provenance(self.graph)
        frozen = self.node.meta["from_node"][0]
        exported = frozen.to_dict()
        exported["from_node"][0]["pass_name"] = "changed"
        exported["from_node"].clear()
        self.assertEqual(frozen.to_dict(), expected)
        cached = frozen._dict
        with patch.object(
            _ImmutableNodeSource,
            "_get_action_string",
            side_effect=AssertionError("rebuilt cached provenance"),
        ):
            self.assertEqual(frozen.to_dict(), expected)
            self.assertEqual(frozen, self.source)
            self.assertEqual(hash(frozen), hash(self.source))
        self.assertIs(frozen._dict, cached)
        expected_hash = hash(self.source)
        with patch.object(
            NodeSource, "__hash__", side_effect=AssertionError("recomputed cached hash")
        ):
            self.assertEqual(hash(frozen), expected_hash)

    def test_torch_node_source_compatibility(self):
        self.assertEqual(
            set(vars(NodeSource(None))),
            {
                "pass_name",
                "action",
                "from_node",
                "node_info",
                "_dict",
                "_action_string",
            },
        )
        freeze_node_provenance(self.graph)
        frozen = self.node.meta["from_node"][0]
        self.assertEqual(frozen, self.source)
        self.assertEqual(self.source, frozen)
        self.assertEqual(hash(frozen), hash(self.source))
        self.assertEqual(repr(frozen), repr(self.source))
        self.assertEqual(frozen.print_readable(), self.source.print_readable())
        self.assertEqual(
            (frozen.name, frozen.target, frozen.graph_id),
            (self.source.name, self.source.target, self.source.graph_id),
        )
        self.assertNotEqual(frozen, object())

    def test_pickle_restores_standard_torch_node_sources(self):
        expected = self.source.to_dict()
        freeze_node_provenance(self.graph)
        frozen = self.node.meta["from_node"][0]
        serialized = pickle.dumps(frozen)
        self.assertNotIn(b"executorch", serialized)
        restored = pickle.loads(serialized)
        self.assertIs(type(restored), NodeSource)
        self.assertIs(type(restored.from_node[0]), NodeSource)
        self.assertEqual(restored.to_dict(), expected)
        restored.action.append(NodeSourceAction.CREATE)
        self.assertEqual(frozen.to_dict(), expected)

    def test_freezing_is_idempotent(self):
        freeze_node_provenance(self.graph)
        sources = self.node.meta["from_node"]
        freeze_node_provenance(self.graph)
        self.assertIs(self.node.meta["from_node"], sources)

    def test_non_node_source_entries_are_preserved(self):
        entries = [("source", "convolution"), {"custom": []}, None]
        self.node.meta["from_node"] = [self.source, *entries]
        freeze_node_provenance(self.graph)
        self.assertIsInstance(self.node.meta["from_node"][0], _ImmutableNodeSource)
        for actual, expected in zip(self.node.meta["from_node"][1:], entries):
            self.assertIs(actual, expected)

    def test_unsupported_ancestry_is_not_shared_by_deepcopy(self):
        custom = {"custom": []}
        self.source.from_node = [self.ancestor, ("source", "convolution"), custom]
        sources = self.node.meta["from_node"]
        freeze_node_provenance(self.graph)
        self.assertIs(self.node.meta["from_node"], sources)
        self.assertIs(sources[0], self.source)
        copied = copy.deepcopy(sources)
        copied[0].from_node[2]["custom"].append("changed")
        self.assertEqual(custom, {"custom": []})

    def test_node_source_subclasses_are_preserved(self):
        class CustomNodeSource(NodeSource):
            pass

        source = CustomNodeSource(None)
        self.node.meta["from_node"] = [source]
        freeze_node_provenance(self.graph)
        self.assertIs(self.node.meta["from_node"][0], source)

    def test_node_source_without_node_info(self):
        source = NodeSource(None)
        self.node.meta["from_node"] = [source]
        freeze_node_provenance(self.graph)
        frozen = self.node.meta["from_node"][0]
        self.assertEqual(frozen, source)
        self.assertEqual(pickle.loads(pickle.dumps(frozen)), source)

    def test_first_replay_does_not_deepcopy_mutable_node_sources(self):
        inputs = (torch.randn(2, 3),)
        exported = torch.export.export(Model(), inputs, strict=True)
        source_node = next(
            node for node in exported.graph.nodes if node.op == "call_function"
        )
        source = NodeSource(source_node, "original", NodeSourceAction.CREATE)
        source_node.meta["from_node"] = [source]
        original_meta = source_node.meta
        original_sources = source_node.meta["from_node"]
        expected = source.to_dict()

        with patch.object(
            NodeSource,
            "__deepcopy__",
            side_effect=AssertionError("deepcopy traversed mutable input provenance"),
            create=True,
        ):
            result = ExportPass()(exported.graph_module).graph_module

        self.assertIs(source_node.meta, original_meta)
        self.assertIs(source_node.meta["from_node"], original_sources)
        self.assertIs(type(original_sources[0]), NodeSource)
        self.assertEqual(source.to_dict(), expected)
        result_node = next(
            node for node in result.graph.nodes if node.op == "call_function"
        )
        self.assertIsInstance(result_node.meta["from_node"][0], _ImmutableNodeSource)
        self.assertEqual(result_node.meta["from_node"][0].to_dict(), expected)

    def test_replay_preserves_ancestry_without_mutating_input(self):
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
        original_sources = source_node.meta["from_node"]
        for _ in range(4):
            graph_module = ExportPass()(graph_module).graph_module

        self.assertIs(source_node.meta["from_node"], original_sources)
        self.assertIs(type(original_sources[0]), NodeSource)
        self.assertIsInstance(original_sources[0].action, list)
        self.assertEqual(original_sources[0].to_dict(), expected_history)
        output_node = next(
            node for node in graph_module.graph.nodes if node.op == "call_function"
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
        branch_node = next(
            node for node in branch.graph.nodes if node.op == "call_function"
        )
        self.assertIs(
            branch_node.meta["from_node"][0], output_node.meta["from_node"][0]
        )

    def test_export_round_trip(self):
        inputs = (torch.randn(2, 3),)
        ep = torch.export.export(Model(), inputs, strict=True)
        source_node = next(
            node for node in ep.graph.nodes if node.op == "call_function"
        )
        source_node.meta["from_node"] = [
            NodeSource(source_node, "original", NodeSourceAction.CREATE)
        ]
        freeze_node_provenance(ep.graph)
        expected = [source.to_dict() for source in source_node.meta["from_node"]]
        buffer = io.BytesIO()
        torch.export.save(ep, buffer)
        buffer.seek(0)
        loaded = torch.export.load(buffer)
        loaded_node = next(
            node for node in loaded.graph.nodes if node.op == "call_function"
        )
        self.assertEqual(
            [source.to_dict() for source in loaded_node.meta["from_node"]], expected
        )
        torch.testing.assert_close(loaded.module()(*inputs), Model()(*inputs))

    def _assert_frozen_provenance(self, graph_module):
        for module in graph_module.modules():
            if isinstance(module, torch.fx.GraphModule):
                for node in module.graph.nodes:
                    for source in node.meta.get("from_node", []):
                        self.assertIsInstance(source, _ImmutableNodeSource)

    def test_control_flow_submodules_preserve_input_metadata(self):
        class CondModel(torch.nn.Module):
            def forward(self, inputs):
                return torch.cond(
                    inputs.sum() > 0,
                    lambda values: values.sin(),
                    lambda values: values.cos(),
                    (inputs,),
                )

        class MapModel(torch.nn.Module):
            def forward(self, inputs):
                return torch_map(lambda values: values.sin(), inputs)

        for model in (CondModel(), MapModel()):
            with self.subTest(model=type(model).__name__):
                inputs = (torch.randn(2, 3),)
                ep = torch.export.export(model, inputs, strict=True)
                originals = []
                for module in ep.graph_module.modules():
                    if not isinstance(module, torch.fx.GraphModule):
                        continue
                    for node in module.graph.nodes:
                        if node.op == "call_function":
                            sources = [
                                NodeSource(node, "original", NodeSourceAction.CREATE)
                            ]
                            node.meta["from_node"] = sources
                            originals.append((node, sources, sources[0].to_dict()))
                result = ExportPass()(ep.graph_module).graph_module
                result = ExportPass()(result).graph_module
                self.assertGreater(len(originals), 1)
                for node, sources, expected in originals:
                    self.assertIs(node.meta["from_node"], sources)
                    self.assertIs(type(sources[0]), NodeSource)
                    self.assertEqual(sources[0].to_dict(), expected)
                self._assert_frozen_provenance(result)
                for values in (torch.ones(2, 3), -torch.ones(2, 3)):
                    torch.testing.assert_close(result(values), ep.graph_module(values))


if __name__ == "__main__":
    unittest.main()
