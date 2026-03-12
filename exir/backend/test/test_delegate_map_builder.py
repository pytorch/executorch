# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Iterator, Union

import torch
from executorch import exir
from executorch.exir import to_edge
from executorch.exir.backend.backend_api import to_backend
from torch.export import export
from executorch.exir.backend.test.backend_with_delegate_mapping_demo import (
    BackendWithDelegateMappingDemo,
)

from executorch.exir.backend.utils import DelegateMappingBuilder


class TestDelegateMapBuilder(unittest.TestCase):
    def setUp(self) -> None:
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                y = torch.sin(x)
                return torch.cos(y)

        model = Model()
        model_inputs = (torch.ones(1, 1),)

        program = (
            to_edge(export(model, model_inputs, strict=True))
            .to_executorch()
        )

        # Create nodes for testing mapping
        self.nodes = list(program.exported_program().graph_module.graph.nodes)

        self.handles = [node.meta.get("debug_handle") for node in self.nodes]
        # Extract the actual debug handle values for sin and cos nodes
        non_none_handles = [h for h in self.handles if h is not None]
        self.sin_handle = non_none_handles[0]
        self.cos_handle = non_none_handles[1]
        # Find the index of the sin node in self.nodes
        self.sin_node_idx = next(
            i for i, n in enumerate(self.nodes) if n.meta.get("debug_handle") == self.sin_handle
        )
        self.cos_node_idx = next(
            i for i, n in enumerate(self.nodes) if n.meta.get("debug_handle") == self.cos_handle
        )

    def test_basic_generated_identifier(self):
        delegate_builder = DelegateMappingBuilder(generated_identifiers=True)
        sh, ch = self.sin_handle, self.cos_handle

        expected_mapping = {0: (sh, ch)}
        self.assertEqual(
            delegate_builder.insert_delegate_mapping_entry(nodes=self.nodes), 0
        )
        self.assertEqual(delegate_builder.get_delegate_mapping(), expected_mapping)

        expected_mapping = {0: (sh, ch), 1: (sh,)}
        self.assertEqual(
            delegate_builder.insert_delegate_mapping_entry(nodes=self.nodes[self.sin_node_idx]), 1
        )
        self.assertEqual(delegate_builder.get_delegate_mapping(), expected_mapping)

        expected_mapping = {0: (sh, ch), 1: (sh,), 2: (ch,)}
        self.assertEqual(
            delegate_builder.insert_delegate_mapping_entry(handles=self.handles[self.cos_node_idx]),
            2,
        )
        self.assertEqual(delegate_builder.get_delegate_mapping(), expected_mapping)

        expected_mapping = {
            0: (sh, ch),
            1: (sh,),
            2: (ch,),
            3: (sh, ch),
        }
        self.assertEqual(
            delegate_builder.insert_delegate_mapping_entry(handles=self.handles), 3
        )
        self.assertEqual(delegate_builder.get_delegate_mapping(), expected_mapping)

    def test_basic_manual_int_identifier(self):
        self._test_basic_manual_identifier(iter([22, 55]))

    def test_basic_manual_string_identifier(self):
        self._test_basic_manual_identifier(iter(["22", "55"]))

    def test_adding_manual_identifier_when_generated(self):
        delegate_builder = DelegateMappingBuilder(generated_identifiers=True)

        self.assertRaises(
            Exception,
            lambda: delegate_builder.insert_delegate_mapping_entry(
                nodes=self.nodes, identifier="22"
            ),
        )
        self.assertRaises(
            Exception,
            lambda: delegate_builder.insert_delegate_mapping_entry(
                handles=self.handles, identifier="22"
            ),
        )

    def test_omitting_identifier_when_not_generated(self):
        delegate_builder = DelegateMappingBuilder()

        self.assertRaises(
            Exception,
            lambda: delegate_builder.insert_delegate_mapping_entry(nodes=self.nodes),
        )
        self.assertRaises(
            Exception,
            lambda: delegate_builder.insert_delegate_mapping_entry(
                handles=self.handles
            ),
        )

    def test_reinsert_delegate_debug_identifier(self):
        delegate_builder = DelegateMappingBuilder()
        delegate_builder.insert_delegate_mapping_entry(
            nodes=self.nodes[self.sin_node_idx], identifier="1"
        )

        self.assertRaises(
            Exception,
            lambda: delegate_builder.insert_delegate_mapping_entry(
                nodes=self.nodes[0], identifier="1"
            ),
        )
        self.assertRaises(
            Exception,
            lambda: delegate_builder.insert_delegate_mapping_entry(
                handles=self.handles[0], identifier="1"
            ),
        )

        self.assertRaises(
            Exception,
            lambda: delegate_builder.insert_delegate_mapping_entry(
                nodes=self.nodes[self.sin_node_idx], identifier="1"
            ),
        )
        self.assertRaises(
            Exception,
            lambda: delegate_builder.insert_delegate_mapping_entry(
                handles=self.handles[self.sin_node_idx], identifier="1"
            ),
        )

    def test_backend_with_delegate_mapping(self) -> None:
        model, inputs = BackendWithDelegateMappingDemo.get_test_model_and_inputs()
        edgeir_m = to_edge(
            export(model, inputs, strict=True),
            compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
        )
        lowered_module = to_backend(
            "BackendWithDelegateMappingDemo", edgeir_m.exported_program(), []
        )
        debug_handle_map = lowered_module.meta.get("debug_handle_map")
        self.assertIsNotNone(debug_handle_map)
        # There should be 3 backend ops in this model.
        self.assertEqual(len(debug_handle_map), 5)
        # Check to see that all the delegate debug indexes in the range [0,2] are present.
        self.assertTrue(
            all(element in debug_handle_map.keys() for element in [1, 2, 3, 4])
        )

        class CompositeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lowered_module = lowered_module

            def forward(self, x):
                return self.lowered_module(x)

        composite_model = CompositeModule()
        # TODO: Switch this to lowered_module.program() once lowered_module has support
        # for storing debug delegate identifier maps.
        to_edge(export(composite_model, inputs, strict=True)).to_executorch()

    def test_passing_both_nodes_and_handles(self):
        delegate_builder = DelegateMappingBuilder()

        self.assertRaises(
            Exception,
            lambda: delegate_builder.insert_delegate_mapping_entry(
                nodes=self.nodes, handles=self.handles
            ),
        )

    def test_missing_handle_filtering(self):
        delegate_builder = DelegateMappingBuilder()
        self.assertRaises(
            Exception,
            lambda: delegate_builder.insert_delegate_mapping_entry(handles=[None]),
        )
        self.assertRaises(
            Exception,
            lambda: delegate_builder.insert_delegate_mapping_entry(nodes=[None]),
        )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _test_basic_manual_identifier(self, identifiers: Iterator[Union[int, str]]):
        """
        Using the iteration of identifiers:
        1) Create a Delegate Map Builder
        2) Add an entry with a list of Nodes using the first identifier
        3) Add an entry with a single node using the second identifier

        Verify behavior results
        """

        delegate_builder_nodes = DelegateMappingBuilder()
        delegate_builder_handles = DelegateMappingBuilder()
        sh, ch = self.sin_handle, self.cos_handle

        # Entry with a list of nodes
        iden_1 = next(identifiers)
        expected_mapping = {iden_1: (sh, ch)}
        self.assertEqual(
            delegate_builder_nodes.insert_delegate_mapping_entry(
                nodes=self.nodes, identifier=iden_1
            ),
            iden_1,
        )
        self.assertEqual(
            delegate_builder_handles.insert_delegate_mapping_entry(
                handles=self.handles, identifier=iden_1
            ),
            iden_1,
        )
        self.assertEqual(
            delegate_builder_nodes.get_delegate_mapping(), expected_mapping
        )
        self.assertEqual(
            delegate_builder_handles.get_delegate_mapping(), expected_mapping
        )

        # Entry with a single node
        iden_2 = next(identifiers)
        expected_mapping = {iden_1: (sh, ch), iden_2: (sh,)}
        self.assertEqual(
            delegate_builder_nodes.insert_delegate_mapping_entry(
                nodes=self.nodes[self.sin_node_idx], identifier=iden_2
            ),
            iden_2,
        )
        self.assertEqual(
            delegate_builder_handles.insert_delegate_mapping_entry(
                handles=self.handles[self.sin_node_idx], identifier=iden_2
            ),
            iden_2,
        )
        self.assertEqual(
            delegate_builder_nodes.get_delegate_mapping(), expected_mapping
        )
        self.assertEqual(
            delegate_builder_handles.get_delegate_mapping(), expected_mapping
        )
