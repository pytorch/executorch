# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Iterator, Union

import torch
from executorch import exir
from executorch.exir.backend.backend_api import to_backend
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
            exir.capture(model, model_inputs, exir.CaptureConfig(pt2_mode=True))
            .to_edge()
            .to_executorch()
        )

        # Create nodes for testing mapping
        # nodes: [arg0_1, alloc, aten_sin_default, alloc_1, aten_cos_default, output]
        # debug handles: [0, None, 1, None, 2, 3]
        self.nodes = list(program.graph_module.graph.nodes)

        self.handles = [node.meta.get("debug_handle") for node in self.nodes]

    def test_basic_generated_identifier(self):
        delegate_builder = DelegateMappingBuilder(generated_identifiers=True)

        expected_mapping = {0: (0, 1, 2, 3)}
        self.assertEqual(
            delegate_builder.insert_delegate_mapping_entry(nodes=self.nodes), 0
        )
        self.assertEqual(delegate_builder.get_delegate_mapping(), expected_mapping)

        expected_mapping = {0: (0, 1, 2, 3), 1: (0,)}
        self.assertEqual(
            delegate_builder.insert_delegate_mapping_entry(nodes=self.nodes[0]), 1
        )
        self.assertEqual(delegate_builder.get_delegate_mapping(), expected_mapping)

        expected_mapping = {0: (0, 1, 2, 3), 1: (0,), 2: (1,)}
        self.assertEqual(
            delegate_builder.insert_delegate_mapping_entry(handles=self.handles[2]),
            2,
        )
        self.assertEqual(delegate_builder.get_delegate_mapping(), expected_mapping)

        expected_mapping = {
            0: (0, 1, 2, 3),
            1: (0,),
            2: (1,),
            3: (0, 1, 2, 3),
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
            nodes=self.nodes[0], identifier="1"
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

    def test_backend_with_delegate_mapping(self) -> None:
        model, inputs = BackendWithDelegateMappingDemo.get_test_model_and_inputs()
        edgeir_m = exir.capture(model, inputs, exir.CaptureConfig()).to_edge(
            exir.EdgeCompileConfig(_check_ir_validity=False)
        )
        lowered_module = to_backend(
            "BackendWithDelegateMappingDemo", edgeir_m.exported_program, []
        )
        debug_handle_map = lowered_module.meta.get("debug_handle_map")
        self.assertIsNotNone(debug_handle_map)
        # There should be 3 backend ops in this model.
        self.assertEqual(len(debug_handle_map), 5)
        # Check to see that all the delegate debug indexes in the range [0,2] are present.
        self.assertTrue(
            all(element in debug_handle_map.keys() for element in [0, 1, 2, 3])
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
        exir.capture(
            composite_model, inputs, exir.CaptureConfig()
        ).to_edge().to_executorch()

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

        # Entry with a list of nodes
        iden_1 = next(identifiers)
        expected_mapping = {iden_1: (0, 1, 2, 3)}
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
        expected_mapping = {iden_1: (0, 1, 2, 3), iden_2: (0,)}
        self.assertEqual(
            delegate_builder_nodes.insert_delegate_mapping_entry(
                nodes=self.nodes[0], identifier=iden_2
            ),
            iden_2,
        )
        self.assertEqual(
            delegate_builder_handles.insert_delegate_mapping_entry(
                handles=self.handles[0], identifier=iden_2
            ),
            iden_2,
        )
        self.assertEqual(
            delegate_builder_nodes.get_delegate_mapping(), expected_mapping
        )
        self.assertEqual(
            delegate_builder_handles.get_delegate_mapping(), expected_mapping
        )
