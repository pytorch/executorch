# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.vulkan.serialization.vulkan_graph_builder import VkGraphBuilder
from executorch.backends.vulkan.serialization.vulkan_graph_schema import (
    Bool,
    BoolList,
    Double,
    Int,
    IntList,
)
from executorch.backends.vulkan.vulkan_preprocess import apply_passes
from executorch.exir import to_edge
from executorch.exir.backend.utils import DelegateMappingBuilder
from executorch.exir.passes import SpecPropPass


class TestVkGraphBuilderScalarTensor(unittest.TestCase):
    def test_scalar_cache_preserves_types_and_signed_zero(self):
        program = torch.export.export(torch.nn.Identity(), (torch.ones(1),))
        builder = VkGraphBuilder(
            program, DelegateMappingBuilder(generated_identifiers=True)
        )
        scalars = (1.0, 1, True, 0.0, -0.0, 0, False)
        expected = (
            Double(1.0),
            Int(1),
            Bool(True),
            Double(0.0),
            Double(-0.0),
            Int(0),
            Bool(False),
        )
        ids = [builder.get_or_create_scalar_value(value) for value in scalars]
        self.assertEqual(len(set(ids)), len(scalars))
        for value, value_id, serialized in zip(scalars, ids, expected):
            self.assertEqual(builder.get_or_create_scalar_value(value), value_id)
            self.assertEqual(repr(builder.values[value_id].value), repr(serialized))

    def test_aten_scalar_tensor_keeps_namespace(self):
        class Mask(torch.nn.Module):
            def forward(self, x):
                return torch.where(x, 0.0, -torch.inf)

        program = torch.export.export(Mask(), (torch.tensor([True, False]),))
        edge = to_edge(program)
        program = apply_passes(edge.exported_program(), [SpecPropPass()])
        self.assertEqual(
            sum(
                node.target == torch.ops.aten.scalar_tensor.default
                for node in program.graph.nodes
            ),
            2,
        )
        graph = VkGraphBuilder(
            program, DelegateMappingBuilder(generated_identifiers=True)
        ).build_graph()
        names = [op.name for op in graph.chain]
        self.assertEqual(names.count("aten.scalar_tensor.default"), 2)
        self.assertNotIn("scalar_tensor.default", names)


class TestVkGraphBuilderInputIds(unittest.TestCase):
    """The serialized input list has to match the delegate call's arguments.

    VulkanBackend::execute walks `args` positionally against
    ComputeGraph::inputs() and rejects the call when the counts disagree, so
    every placeholder that the delegate call passes must appear in input_ids,
    including ones this graph happens not to use. Unused placeholders are not
    hypothetical: passes that run after partitioning can fold away a
    placeholder's only consumers, leaving the argument list and the serialized
    graph out of step.
    """

    def _build(self, module: torch.nn.Module, inputs) -> VkGraphBuilder:
        edge = to_edge(torch.export.export(module, inputs, strict=True))
        # The builder reads node specs, which the backend's own preprocess
        # populates before it gets here.
        program = apply_passes(edge.exported_program(), [SpecPropPass()])
        builder = VkGraphBuilder(
            program, DelegateMappingBuilder(generated_identifiers=True)
        )
        builder.build_graph()
        return builder

    def test_unused_placeholder_is_still_declared_as_an_input(self) -> None:
        class UsesOnlyTheFirstInput(torch.nn.Module):
            def forward(self, used, unused):
                return used + used

        builder = self._build(
            UsesOnlyTheFirstInput(), (torch.randn(2, 3), torch.randn(2, 3))
        )
        self.assertEqual(len(builder.input_ids), 2)

    def test_used_placeholders_are_declared_in_order(self) -> None:
        class UsesBothInputs(torch.nn.Module):
            def forward(self, first, second):
                return first + second

        builder = self._build(UsesBothInputs(), (torch.randn(2, 3), torch.randn(2, 3)))
        self.assertEqual(len(builder.input_ids), 2)


class TestVkGraphBuilderScalarLists(unittest.TestCase):
    """A scalar list serializes to exactly one Value. bool is a subclass of int,
    so a bool list also satisfies the all-int check, and an empty list
    satisfies every check."""

    def _values_appended_by(self, arg) -> list:
        program = torch.export.export(torch.nn.Identity(), (torch.ones(1),))
        builder = VkGraphBuilder(
            program, DelegateMappingBuilder(generated_identifiers=True)
        )
        value_id = builder.create_scalar_list_value(arg)
        self.assertEqual(value_id, 0)
        return [v.value for v in builder.values]

    def test_bool_list_serializes_once(self) -> None:
        values = self._values_appended_by([True, False])
        self.assertEqual(len(values), 1)
        self.assertIsInstance(values[0], BoolList)

    def test_empty_list_serializes_once(self) -> None:
        values = self._values_appended_by([])
        self.assertEqual(len(values), 1)
        self.assertIsInstance(values[0], IntList)

    def test_int_list_serializes_once(self) -> None:
        values = self._values_appended_by([1, 2])
        self.assertEqual(len(values), 1)
        self.assertIsInstance(values[0], IntList)

    def test_mixed_bool_and_int_list_serializes_as_ints(self) -> None:
        values = self._values_appended_by([True, 2])
        self.assertEqual(len(values), 1)
        self.assertIsInstance(values[0], IntList)
        self.assertEqual(values[0].items, [1, 2])
        self.assertTrue(all(type(e) is int for e in values[0].items))


if __name__ == "__main__":
    unittest.main()
