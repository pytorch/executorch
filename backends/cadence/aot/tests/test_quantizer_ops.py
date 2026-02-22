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
from executorch.backends.cadence.aot.graph_builder import (
    GraphBuilder,
    single_op_builder,
)
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
    qconfig_A8W8sym,
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
# For fused patterns (e.g., conv+relu), an optional third element specifies the node
# whose args contain the quantized inputs (e.g., conv node for conv+relu fusion).
GraphBuilderFn = Callable[
    ["QuantizerAnnotationTest"],
    tuple[torch.fx.GraphModule, torch.fx.Node]
    | tuple[torch.fx.GraphModule, torch.fx.Node, torch.fx.Node],
]


# Quantizers intentionally excluded from annotation testing.
# These should be explicitly justified when added.
EXCLUDED_FROM_ANNOTATION_TESTING: set[type[CadenceQuantizer]] = {
    CadenceNopQuantizer,  # No-op quantizer, doesn't annotate anything
    CadenceW8A32MixedQuantizer,  # TODO: T247438158 Add test coverage
    CadenceRmsNormNopQuantizer,  # No-op quantizer, doesn't annotate anything, preserves rms_norm from decomposition
}


# Test case definitions for quantizer annotation tests.
# Format: (name, graph_builder_fn, quantizer_instance, target_op, expected_output_qspec, expected_input_qspecs)
# Adding a new quantizer test only requires adding a tuple to this list.
# Note: Use None in expected_input_qspecs to skip comparison for that input (e.g., for DerivedQuantizationSpec).
QUANTIZER_ANNOTATION_TEST_CASES: list[
    tuple[
        str,
        GraphBuilderFn,
        CadenceQuantizer,
        OpOverload,
        QuantizationSpec,
        list[QuantizationSpec | None],
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
    (
        "conv1d_A16",
        lambda self: self._build_conv1d_graph(),
        CadenceWith16BitConvActivationsQuantizer(),
        torch.ops.aten.conv1d.default,
        qconfig_A16.output_activation,
        # For conv1d: [input_activation, weight]
        [qconfig_A16.input_activation, qconfig_A16.weight],
    ),
    (
        "conv2d_A16",
        lambda self: self._build_conv2d_graph(),
        CadenceWith16BitConvActivationsQuantizer(),
        torch.ops.aten.conv2d.default,
        qconfig_A16.output_activation,
        # For conv2d: [input_activation, weight]
        [qconfig_A16.input_activation, qconfig_A16.weight],
    ),
    (
        "softmax_A16",
        lambda self: self._build_softmax_graph(),
        CadenceWithSoftmaxQuantizer(),
        torch.ops.aten._softmax.default,
        qconfig_A16.output_activation,
        # For softmax: only input_activation
        [qconfig_A16.input_activation],
    ),
    (
        "layer_norm_A8W8",
        lambda self: self._build_layer_norm_graph(),
        CadenceWithLayerNormQuantizer(),
        torch.ops.aten.layer_norm.default,
        qconfig_A8W8.output_activation,
        # For layer_norm: only input_activation (weights/bias are passed as others)
        [qconfig_A8W8.input_activation],
    ),
    (
        "add_A8W8",
        lambda self: self._build_add_graph(),
        CadenceWakeWordQuantizer(),
        torch.ops.aten.add.Tensor,
        qconfig_A8W8.output_activation,
        # For add: both inputs are activations
        [qconfig_A8W8.input_activation, qconfig_A8W8.input_activation],
    ),
    # CadenceDefaultQuantizer test cases
    (
        "default_matmul_A8W8",
        lambda self: self._build_matmul_graph(),
        CadenceDefaultQuantizer(),
        torch.ops.aten.matmul.default,
        qconfig_A8W8.output_activation,
        # For matmul: both inputs are activations
        [qconfig_A8W8.input_activation, qconfig_A8W8.input_activation],
    ),
    (
        "default_linear_A8W8",
        lambda self: self._build_linear_graph(),
        CadenceDefaultQuantizer(),
        torch.ops.aten.linear.default,
        qconfig_A8W8.output_activation,
        # For linear: [input_activation, weight]
        [qconfig_A8W8.input_activation, qconfig_A8W8.weight],
    ),
    (
        "default_conv1d_A8W8sym",
        lambda self: self._build_conv1d_graph(),
        CadenceDefaultQuantizer(),
        torch.ops.aten.conv1d.default,
        qconfig_A8W8sym.output_activation,
        # For conv1d: [input_activation, weight] with symmetric weights
        [qconfig_A8W8sym.input_activation, qconfig_A8W8sym.weight],
    ),
    (
        "default_conv2d_A8W8sym",
        lambda self: self._build_conv2d_graph(),
        CadenceDefaultQuantizer(),
        torch.ops.aten.conv2d.default,
        qconfig_A8W8sym.output_activation,
        # For conv2d: [input_activation, weight] with symmetric weights
        [qconfig_A8W8sym.input_activation, qconfig_A8W8sym.weight],
    ),
    (
        "default_bmm_A8W8",
        lambda self: self._build_bmm_graph(),
        CadenceDefaultQuantizer(),
        torch.ops.aten.bmm.default,
        qconfig_A8W8.output_activation,
        # For bmm: both inputs are activations
        [qconfig_A8W8.input_activation, qconfig_A8W8.input_activation],
    ),
    (
        "default_relu_A8W8",
        lambda self: self._build_relu_graph(),
        CadenceDefaultQuantizer(),
        torch.ops.aten.relu.default,
        qconfig_A8W8.output_activation,
        # For relu: only input_activation
        [qconfig_A8W8.input_activation],
    ),
    (
        "default_addmm_A8W8",
        lambda self: self._build_addmm_graph(),
        CadenceDefaultQuantizer(),
        torch.ops.aten.addmm.default,
        qconfig_A8W8.output_activation,
        # For addmm: [bias (DerivedQuantizationSpec), mat1, mat2]
        # Use None to skip comparison for bias since it's a DerivedQuantizationSpec
        [None, qconfig_A8W8.input_activation, qconfig_A8W8.weight],
    ),
    # CadenceFusedConvReluQuantizer test cases
    (
        "fused_conv2d_relu_A8W8sym",
        lambda self: self._build_conv2d_relu_graph(),
        CadenceFusedConvReluQuantizer(),
        torch.ops.aten.relu.default,
        qconfig_A8W8sym.output_activation,
        # For fused conv2d+relu: [input_activation, weight] from conv2d node
        [qconfig_A8W8sym.input_activation, qconfig_A8W8sym.weight],
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

    def _build_conv1d_graph(self) -> tuple[torch.fx.GraphModule, torch.fx.Node]:
        """Build a simple graph with a conv1d operation (no bias)."""
        builder = GraphBuilder()
        # Input shape: (batch, in_channels, length)
        x = builder.placeholder("x", torch.randn(1, 3, 10))
        # Weight shape: (out_channels, in_channels, kernel_size)
        weight = builder.placeholder("weight", torch.randn(6, 3, 3))
        conv1d = builder.call_operator(
            op=torch.ops.aten.conv1d.default,
            args=(x, weight),
            meta=NodeMetadata(
                {"source_fn_stack": [("conv1d", torch.ops.aten.conv1d.default)]}
            ),
        )
        builder.output([conv1d])
        gm = builder.get_graph_module()

        conv1d_nodes = gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.aten.conv1d.default,
        )
        self.assertEqual(len(conv1d_nodes), 1, "Should find exactly one conv1d node")
        return gm, conv1d_nodes[0]

    def _build_conv2d_graph(self) -> tuple[torch.fx.GraphModule, torch.fx.Node]:
        """Build a simple graph with a conv2d operation (no bias)."""
        builder = GraphBuilder()
        # Input shape: (batch, in_channels, height, width)
        x = builder.placeholder("x", torch.randn(1, 3, 8, 8))
        # Weight shape: (out_channels, in_channels, kernel_h, kernel_w)
        weight = builder.placeholder("weight", torch.randn(6, 3, 3, 3))
        conv2d = builder.call_operator(
            op=torch.ops.aten.conv2d.default,
            args=(x, weight),
            meta=NodeMetadata(
                {"source_fn_stack": [("conv2d", torch.ops.aten.conv2d.default)]}
            ),
        )
        builder.output([conv2d])
        gm = builder.get_graph_module()

        conv2d_nodes = gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.aten.conv2d.default,
        )
        self.assertEqual(len(conv2d_nodes), 1, "Should find exactly one conv2d node")
        return gm, conv2d_nodes[0]

    def _build_softmax_graph(self) -> tuple[torch.fx.GraphModule, torch.fx.Node]:
        """Build a simple graph with a softmax operation."""
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 10))
        softmax = builder.call_operator(
            op=torch.ops.aten._softmax.default,
            args=(x, -1, False),  # dim=-1, half_to_float=False
            meta=NodeMetadata(
                {"source_fn_stack": [("softmax", torch.ops.aten._softmax.default)]}
            ),
        )
        builder.output([softmax])
        gm = builder.get_graph_module()

        softmax_nodes = gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.aten._softmax.default,
        )
        self.assertEqual(len(softmax_nodes), 1, "Should find exactly one softmax node")
        return gm, softmax_nodes[0]

    def _build_layer_norm_graph(self) -> tuple[torch.fx.GraphModule, torch.fx.Node]:
        """Build a simple graph with a layer_norm operation."""
        # Input shape: (batch, features)
        x = torch.randn(1, 10)
        # normalized_shape must match the last dimension(s) of input
        normalized_shape = [10]
        gm = single_op_builder(
            placeholders=(x,),
            op=torch.ops.aten.layer_norm.default,
            args=(x, normalized_shape),
        )

        layer_norm_nodes = gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.aten.layer_norm.default,
        )
        self.assertEqual(
            len(layer_norm_nodes), 1, "Should find exactly one layer_norm node"
        )
        # Add source_fn_stack metadata required by quantizer pattern matching
        layer_norm_nodes[0].meta["source_fn_stack"] = [
            ("layer_norm", torch.ops.aten.layer_norm.default)
        ]
        return gm, layer_norm_nodes[0]

    def _build_add_graph(self) -> tuple[torch.fx.GraphModule, torch.fx.Node]:
        """Build a simple graph with an add operation."""
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 10))
        y = builder.placeholder("y", torch.randn(1, 10))
        add = builder.call_operator(
            op=torch.ops.aten.add.Tensor,
            args=(x, y),
            meta=NodeMetadata(
                {"source_fn_stack": [("add", torch.ops.aten.add.Tensor)]}
            ),
        )
        builder.output([add])
        gm = builder.get_graph_module()

        add_nodes = gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.aten.add.Tensor,
        )
        self.assertEqual(len(add_nodes), 1, "Should find exactly one add node")
        return gm, add_nodes[0]

    def _build_bmm_graph(self) -> tuple[torch.fx.GraphModule, torch.fx.Node]:
        """Build a simple graph with a bmm (batch matrix multiply) operation."""
        builder = GraphBuilder()
        # BMM requires 3D tensors: (batch, n, m) @ (batch, m, p) -> (batch, n, p)
        x = builder.placeholder("x", torch.randn(2, 4, 8))
        y = builder.placeholder("y", torch.randn(2, 8, 4))
        bmm = builder.call_operator(
            op=torch.ops.aten.bmm.default,
            args=(x, y),
            meta=NodeMetadata(
                {"source_fn_stack": [("bmm", torch.ops.aten.bmm.default)]}
            ),
        )
        builder.output([bmm])
        gm = builder.get_graph_module()

        bmm_nodes = gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.aten.bmm.default,
        )
        self.assertEqual(len(bmm_nodes), 1, "Should find exactly one bmm node")
        return gm, bmm_nodes[0]

    def _build_relu_graph(self) -> tuple[torch.fx.GraphModule, torch.fx.Node]:
        """Build a simple graph with a relu operation."""
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 10))
        relu = builder.call_operator(
            op=torch.ops.aten.relu.default,
            args=(x,),
            meta=NodeMetadata(
                {"source_fn_stack": [("relu", torch.ops.aten.relu.default)]}
            ),
        )
        builder.output([relu])
        gm = builder.get_graph_module()

        relu_nodes = gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.aten.relu.default,
        )
        self.assertEqual(len(relu_nodes), 1, "Should find exactly one relu node")
        return gm, relu_nodes[0]

    def _build_addmm_graph(self) -> tuple[torch.fx.GraphModule, torch.fx.Node]:
        """Build a simple graph with an addmm operation."""
        builder = GraphBuilder()
        # addmm: bias + (mat1 @ mat2)
        # args: (bias, mat1, mat2)
        bias = builder.placeholder("bias", torch.randn(5))
        mat1 = builder.placeholder("mat1", torch.randn(1, 10))
        mat2 = builder.placeholder("mat2", torch.randn(10, 5))
        addmm = builder.call_operator(
            op=torch.ops.aten.addmm.default,
            args=(bias, mat1, mat2),
            meta=NodeMetadata(
                {"source_fn_stack": [("addmm", torch.ops.aten.addmm.default)]}
            ),
        )
        builder.output([addmm])
        gm = builder.get_graph_module()

        addmm_nodes = gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.aten.addmm.default,
        )
        self.assertEqual(len(addmm_nodes), 1, "Should find exactly one addmm node")
        return gm, addmm_nodes[0]

    def _build_conv2d_relu_graph(
        self,
    ) -> tuple[torch.fx.GraphModule, torch.fx.Node, torch.fx.Node]:
        """Build a graph with a conv2d followed by relu (fused pattern).

        Returns:
            A tuple of (graph_module, relu_node, conv_node).
            The relu_node is the target node where the annotation is placed.
            The conv_node is the input source node whose args contain the quantized inputs.
        """
        builder = GraphBuilder()
        # Input shape: (batch, in_channels, height, width)
        x = builder.placeholder("x", torch.randn(1, 3, 8, 8))
        # Weight shape: (out_channels, in_channels, kernel_h, kernel_w)
        weight = builder.placeholder("weight", torch.randn(6, 3, 3, 3))
        conv2d = builder.call_operator(
            op=torch.ops.aten.conv2d.default,
            args=(x, weight),
            meta=NodeMetadata(
                {"source_fn_stack": [("conv2d", torch.ops.aten.conv2d.default)]}
            ),
        )
        relu = builder.call_operator(
            op=torch.ops.aten.relu.default,
            args=(conv2d,),
            meta=NodeMetadata(
                {"source_fn_stack": [("relu", torch.ops.aten.relu.default)]}
            ),
        )
        builder.output([relu])
        gm = builder.get_graph_module()

        relu_nodes = gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.aten.relu.default,
        )
        self.assertEqual(len(relu_nodes), 1, "Should find exactly one relu node")

        conv2d_nodes = gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.aten.conv2d.default,
        )
        self.assertEqual(len(conv2d_nodes), 1, "Should find exactly one conv2d node")

        return gm, relu_nodes[0], conv2d_nodes[0]

    @parameterized.expand(QUANTIZER_ANNOTATION_TEST_CASES)
    def test_quantizer_annotation(
        self,
        name: str,
        graph_builder_fn: GraphBuilderFn,
        quantizer: CadenceQuantizer,
        target: OpOverload,
        expected_output_qspec: QuantizationSpec,
        expected_input_qspecs: list[QuantizationSpec | None],
    ) -> None:
        """Parameterized test for quantizer annotations."""
        result = graph_builder_fn(self)
        # Handle both 2-element and 3-element returns from graph builders.
        # For fused patterns, the 3rd element specifies the node whose args
        # contain the quantized inputs (e.g., conv node for conv+relu fusion).
        if len(result) == 3:
            gm = result[0]
            output_node = result[1]
            input_source_node = result[2]
        else:
            gm = result[0]
            output_node = result[1]
            input_source_node = output_node

        quantizer.annotate(gm)

        # Verify output annotation (always on the output node)
        output_annotation: QuantizationAnnotation = output_node.meta[Q_ANNOTATION_KEY]
        self.assertTrue(output_annotation._annotated)
        self.assertEqual(output_annotation.output_qspec, expected_output_qspec)

        # Verify input annotations (on the input source node, which may differ for fused patterns)
        input_annotation: QuantizationAnnotation = input_source_node.meta[
            Q_ANNOTATION_KEY
        ]
        self.assertEqual(
            len(input_annotation.input_qspec_map), len(expected_input_qspecs)
        )
        for input_node, input_qspec in input_annotation.input_qspec_map.items():
            # Find the index of this input node in the input source node's args
            arg_index = None
            args = input_source_node.args
            assert isinstance(args, tuple)
            for i, arg in enumerate(args):
                if arg is input_node:
                    arg_index = i
                    break
            self.assertIsNotNone(
                arg_index,
                f"Input node {input_node} not found in input_source_node.args",
            )
            # Skip comparison if expected qspec is None (e.g., for DerivedQuantizationSpec)
            if expected_input_qspecs[arg_index] is not None:
                self.assertEqual(
                    input_qspec,
                    expected_input_qspecs[arg_index],
                    f"Input qspec mismatch at arg index {arg_index}",
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
