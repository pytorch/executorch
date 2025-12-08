# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from executorch.backends.cadence.aot.graph_builder import GraphBuilder
from executorch.backends.cadence.aot.quantizer.patterns import AddmmPattern

from executorch.backends.cadence.aot.quantizer.quantizer import (
    CadenceAtenQuantizer,
    CadenceDefaultQuantizer,
    CadenceW8A32MixedQuantizer,
    CadenceWith16BitMatmulActivationsQuantizer,
    qconfig_A16,
    qconfig_A8W8,
)
from executorch.exir.pass_base import NodeMetadata
from torchao.quantization.pt2e.quantizer.quantizer import (
    Q_ANNOTATION_KEY,
    QuantizationAnnotation,
)


class QuantizerAnnotationTest(unittest.TestCase):
    """Unit tests for verifying quantizer annotations are correctly applied."""

    def _build_matmul_graph(self) -> tuple[torch.fx.GraphModule, torch.fx.Node]:
        """Build a simple graph with a matmul operation."""
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(4, 8))
        y = builder.placeholder("y", torch.randn(8, 4))
        matmul = builder.call_operator(
            op=torch.ops.aten.matmul.default,
            args=(x, y),
            meta=NodeMetadata(
                {"source_fn_stack": [("matmul", torch.ops.aten.matmul.default)]}
            ),
        )
        builder.output([matmul])
        gm = builder.get_graph_module()

        matmul_nodes = gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.aten.matmul.default,
        )
        self.assertEqual(len(matmul_nodes), 1, "Should find exactly one matmul node")
        return gm, matmul_nodes[0]

    def test_matmul_16bit_quantizer_annotation(self) -> None:
        """Test that CadenceWith16BitMatmulActivationsQuantizer correctly annotates matmul."""
        gm, matmul_node = self._build_matmul_graph()

        quantizer = CadenceWith16BitMatmulActivationsQuantizer()
        quantizer.annotate(gm)

        annotation: QuantizationAnnotation = matmul_node.meta[Q_ANNOTATION_KEY]
        self.assertTrue(annotation._annotated)

        self.assertEqual(annotation.output_qspec, qconfig_A16.output_activation)

        self.assertEqual(len(annotation.input_qspec_map), 2)
        for _, input_qspec in annotation.input_qspec_map.items():
            self.assertEqual(input_qspec, qconfig_A16.input_activation)


class QuantizerOpsPreserveTest(unittest.TestCase):
    def test_mixed_w8a32_ops_to_preserve(self) -> None:
        q = CadenceW8A32MixedQuantizer()
        actual = q.get_ops_to_preserve_from_decomposition()
        expected = [
            torch.ops.aten.linear.default,
            torch.ops.aten.conv1d.default,
            torch.ops.aten.gru.input,
        ]
        self.assertCountEqual(actual, expected)

    def test_default_quantizer_ops_to_preserve(self) -> None:
        q = CadenceDefaultQuantizer()
        actual = q.get_ops_to_preserve_from_decomposition()
        expected = [
            torch.ops.aten.addmm.default,
            torch.ops.aten.bmm.default,
            torch.ops.aten.conv1d.default,
            torch.ops.aten.conv2d.default,
            torch.ops.aten.linear.default,
            torch.ops.aten.matmul.default,
            torch.ops.aten.relu.default,
            torch.ops.aten.relu_.default,
        ]
        self.assertCountEqual(actual, expected)

    def test_nested_quantizer_ops_to_preserve(self) -> None:
        # Setup: Create a nested CadenceQuantizer-like structure by composing
        # - CadenceW8A32MixedQuantizer (which preserves linear, conv1d, gru.input)
        # - A CadenceAtenQuantizer with AddmmPattern (which preserves addmm)
        nested = CadenceDefaultQuantizer(
            quantizers=[
                CadenceW8A32MixedQuantizer(),
                CadenceAtenQuantizer(AddmmPattern(), qconfig_A8W8),
            ]
        )

        # Execute
        actual = nested.get_ops_to_preserve_from_decomposition()

        # Assert: union of both sets without duplicates
        expected = [
            torch.ops.aten.linear.default,
            torch.ops.aten.conv1d.default,
            torch.ops.aten.gru.input,
            torch.ops.aten.addmm.default,
        ]
        self.assertCountEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
