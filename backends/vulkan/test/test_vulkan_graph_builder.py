# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.vulkan.serialization.vulkan_graph_builder import (
    VkGraphBuilder,
)
from executorch.backends.vulkan.vulkan_preprocess import apply_passes
from executorch.exir import to_edge
from executorch.exir.backend.utils import DelegateMappingBuilder
from executorch.exir.passes import SpecPropPass


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


if __name__ == "__main__":
    unittest.main()
