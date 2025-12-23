# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import inspect
import unittest
from typing import Callable

import torch
from executorch.backends.cadence.aot.graph_builder import GraphBuilder
from executorch.backends.cadence.aot.quantizer import quantizer as quantizer_module
from executorch.backends.cadence.aot.quantizer.patterns import AddmmPattern
from executorch.backends.cadence.aot.quantizer.quantizer import (
    CadenceAtenQuantizer,
    CadenceDefaultQuantizer,
    CadenceFusedConvReluQuantizer,
    CadenceNopQuantizer,
    CadenceQuantizer,
    CadenceRmsNormNopQuantizer,
    CadenceW8A32MixedQuantizer,
    CadenceWakeWordQuantizer,
    CadenceWith16BitConvActivationsQuantizer,
    CadenceWith16BitLinearActivationsQuantizer,
    CadenceWith16BitMatmulActivationsQuantizer,
    CadenceWithLayerNormQuantizer,
    CadenceWithSoftmaxQuantizer,
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

# Type alias for graph builder functions.
# These functions take a test instance and return a graph module and the target op node.
GraphBuilderFn = Callable[
    ["QuantizerAnnotationTest"], tuple[torch.fx.GraphModule, torch.fx.Node]
]


# Quantizers intentionally excluded from annotation testing.
# These should be explicitly justified when added.
EXCLUDED_FROM_ANNOTATION_TESTING: set[type[CadenceQuantizer]] = {
    CadenceDefaultQuantizer,  # TODO: T247438143 Add test coverage
    CadenceFusedConvReluQuantizer,  # TODO: T247438151 Add test coverage
    CadenceNopQuantizer,  # No-op quantizer, doesn't annotate anything
    CadenceW8A32MixedQuantizer,  # TODO: T247438158 Add test coverage
    CadenceRmsNormNopQuantizer,  # No-op quantizer, doesn't annotate anything, preserves rms_norm from decomposition
    CadenceWakeWordQuantizer,  # TODO: T247438162 Add test coverage
    CadenceWith16BitConvActivationsQuantizer,  # TODO: T247438221 Add test coverage
    CadenceWithLayerNormQuantizer,  # TODO: T247438410 Add test coverage
    CadenceWithSoftmaxQuantizer,  # TODO: T247438418 Add test coverage
}


# Test case definitions for quantizer annotation tests.
# Format: (name, graph_builder_fn, quantizer_instance, target_op, expected_output_qspec, expected_input_qspecs)
# Adding a new quantizer test only requires adding a tuple to this list.
QUANTIZER_ANNOTATION_TEST_CASES: list[
    tuple[
        str,
        GraphBuilderFn,
        CadenceQuantizer,
        OpOverload,
        QuantizationSpec,
        list[QuantizationSpec],
    ]
] = [
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

# Derive the set of tested quantizer classes from the test cases.
# This ensures TESTED_QUANTIZER_CLASSES stays in sync with actual tests.
TESTED_QUANTIZER_CLASSES: set[type[CadenceQuantizer]] = {
    type(case[2]) for case in QUANTIZER_ANNOTATION_TEST_CASES
}


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

    @parameterized.expand(QUANTIZER_ANNOTATION_TEST_CASES)
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
        self.assertEqual(len(annotation.input_qspec_map), len(expected_input_qspecs))
        for i, (input_node, input_qspec) in enumerate(
            annotation.input_qspec_map.items()
        ):
            expected_arg = op_node.args[i]
            assert isinstance(expected_arg, torch.fx.Node)
            self.assertEqual(
                input_node,
                expected_arg,
                f"Input node mismatch at index {i}",
            )
            self.assertEqual(
                input_qspec,
                expected_input_qspecs[i],
                f"Input qspec mismatch at index {i}",
            )

    def test_all_quantizers_have_annotation_tests(self) -> None:
        """Ensure every CadenceQuantizer subclass is either tested or explicitly excluded."""
        # Get all CadenceQuantizer subclasses defined in the quantizer module
        all_quantizers: set[type[CadenceQuantizer]] = set()
        for _, obj in inspect.getmembers(quantizer_module, inspect.isclass):
            if (
                issubclass(obj, CadenceQuantizer)
                and obj is not CadenceQuantizer
                and obj.__module__ == quantizer_module.__name__
            ):
                all_quantizers.add(obj)

        # Check for missing tests
        untested = (
            all_quantizers - TESTED_QUANTIZER_CLASSES - EXCLUDED_FROM_ANNOTATION_TESTING
        )
        if untested:
            untested_names = sorted(cls.__name__ for cls in untested)
            self.fail(
                f"The following CadenceQuantizer subclasses are not tested in "
                f"test_quantizer_annotation and not in EXCLUDED_FROM_ANNOTATION_TESTING: "
                f"{untested_names}. Please add test cases or explicitly exclude them."
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

    def test_rms_norm_nop_quantizer_ops_to_preserve(self) -> None:
        q = CadenceRmsNormNopQuantizer()
        actual = q.get_ops_to_preserve_from_decomposition()
        expected = [
            torch.ops.aten.rms_norm.default,
        ]
        self.assertCountEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
