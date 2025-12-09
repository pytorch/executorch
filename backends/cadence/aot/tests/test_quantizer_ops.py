# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Callable

import torch
from executorch.backends.cadence.aot.graph_builder import GraphBuilder
from executorch.backends.cadence.aot.quantizer.patterns import AddmmPattern

from executorch.backends.cadence.aot.quantizer.quantizer import (
    CadenceAtenQuantizer,
    CadenceDefaultQuantizer,
    CadenceQuantizer,
    CadenceW8A32MixedQuantizer,
    CadenceWith16BitLinearActivationsQuantizer,
    CadenceWith16BitMatmulActivationsQuantizer,
    qconfig_A16,
    qconfig_A8W8,
)
from executorch.exir.pass_base import NodeMetadata
from parameterized import parameterized
from torch._ops import OpOverload
from torchao.quantization.pt2e.quantizer.quantizer import (
    Q_ANNOTATION_KEY,
    QuantizationAnnotation,
    QuantizationSpec,
)

# Type alias for graph builder functions
GraphBuilderFn = Callable[
    ["QuantizerAnnotationTest"], tuple[torch.fx.GraphModule, torch.fx.Node]
]


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

    def _build_linear_graph(self) -> tuple[torch.fx.GraphModule, torch.fx.Node]:
        """Build a simple graph with a linear operation (no bias)."""
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 10))
        weight = builder.placeholder("weight", torch.randn(5, 10))
        linear = builder.call_operator(
            op=torch.ops.aten.linear.default,
            args=(x, weight),
            meta=NodeMetadata(
                {"source_fn_stack": [("linear", torch.ops.aten.linear.default)]}
            ),
        )
        builder.output([linear])
        gm = builder.get_graph_module()

        linear_nodes = gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.aten.linear.default,
        )
        self.assertEqual(len(linear_nodes), 1, "Should find exactly one linear node")
        return gm, linear_nodes[0]

    @parameterized.expand(
        [
            (
                "matmul_A16",
                lambda self: self._build_matmul_graph(),
                CadenceWith16BitMatmulActivationsQuantizer(),
                torch.ops.aten.matmul.default,
                qconfig_A16.output_activation,
                # For matmul, both inputs are activations
                [qconfig_A16.input_activation, qconfig_A16.input_activation],
            ),
            (
                "linear_A16",
                lambda self: self._build_linear_graph(),
                CadenceWith16BitLinearActivationsQuantizer(),
                torch.ops.aten.linear.default,
                qconfig_A16.output_activation,
                # For linear: [input_activation, weight]
                [qconfig_A16.input_activation, qconfig_A16.weight],
            ),
        ]
    )
    def test_quantizer_annotation(
        self,
        name: str,
        graph_builder_fn: GraphBuilderFn,
        quantizer: CadenceQuantizer,
        target: OpOverload,
        expected_output_qspec: QuantizationSpec,
        expected_input_qspecs: list[QuantizationSpec],
    ) -> None:
        """Parameterized test for quantizer annotations."""
        gm, op_node = graph_builder_fn(self)

        quantizer.annotate(gm)

        annotation: QuantizationAnnotation = op_node.meta[Q_ANNOTATION_KEY]
        self.assertTrue(annotation._annotated)

        # Verify output annotation
        self.assertEqual(annotation.output_qspec, expected_output_qspec)

        # Verify input annotations
        # Build actual_specs in the fixed order defined by op_node.args
        self.assertEqual(len(annotation.input_qspec_map), len(expected_input_qspecs))
        actual_specs = []
        for i in range(len(expected_input_qspecs)):
            arg = op_node.args[i]
            assert isinstance(arg, torch.fx.Node)
            actual_specs.append(annotation.input_qspec_map[arg])

        # Compare expected vs actual specs
        for i, (expected, actual) in enumerate(
            zip(expected_input_qspecs, actual_specs)
        ):
            self.assertEqual(
                actual,
                expected,
                f"Input qspec mismatch at index {i}",
            )


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
