# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for dynamic shapes support in TensorRT backend converters."""

import unittest

import torch
from torch.export import export, Dim


class TestExpandDynamicRegistry(unittest.TestCase):
    """Tests that expand converters declare supports_dynamic_shapes=True."""

    def test_expand_supports_dynamic_shapes(self) -> None:
        from executorch.backends.nvidia.tensorrt.converter_registry import (
            supports_dynamic_shapes,
        )

        # Import converters to trigger registration
        from executorch.backends.nvidia.tensorrt.converters import expand  # noqa: F401

        self.assertTrue(supports_dynamic_shapes("aten.expand.default"))
        self.assertTrue(supports_dynamic_shapes("aten.expand_copy.default"))

    def test_view_supports_dynamic_shapes(self) -> None:
        from executorch.backends.nvidia.tensorrt.converter_registry import (
            supports_dynamic_shapes,
        )

        from executorch.backends.nvidia.tensorrt.converters import reshape  # noqa: F401

        self.assertTrue(supports_dynamic_shapes("aten.view.default"))
        self.assertTrue(supports_dynamic_shapes("aten._unsafe_view.default"))
        self.assertTrue(supports_dynamic_shapes("aten.view_copy.default"))

    def test_slice_supports_dynamic_shapes(self) -> None:
        from executorch.backends.nvidia.tensorrt.converter_registry import (
            supports_dynamic_shapes,
        )

        from executorch.backends.nvidia.tensorrt.converters import slice  # noqa: F401

        self.assertTrue(supports_dynamic_shapes("aten.slice.Tensor"))
        self.assertTrue(supports_dynamic_shapes("aten.slice_copy.Tensor"))

    def test_unsqueeze_supports_dynamic_shapes(self) -> None:
        from executorch.backends.nvidia.tensorrt.converter_registry import (
            supports_dynamic_shapes,
        )

        from executorch.backends.nvidia.tensorrt.converters import reshape  # noqa: F401

        self.assertTrue(supports_dynamic_shapes("aten.unsqueeze.default"))
        self.assertTrue(supports_dynamic_shapes("aten.unsqueeze_copy.default"))

    def test_squeeze_supports_dynamic_shapes(self) -> None:
        from executorch.backends.nvidia.tensorrt.converter_registry import (
            supports_dynamic_shapes,
        )

        from executorch.backends.nvidia.tensorrt.converters import reshape  # noqa: F401

        self.assertTrue(supports_dynamic_shapes("aten.squeeze.dim"))
        self.assertTrue(supports_dynamic_shapes("aten.squeeze.dims"))
        self.assertTrue(supports_dynamic_shapes("aten.squeeze_copy.dim"))
        self.assertTrue(supports_dynamic_shapes("aten.squeeze_copy.dims"))

    def test_flatten_supports_dynamic_shapes(self) -> None:
        from executorch.backends.nvidia.tensorrt.converter_registry import (
            supports_dynamic_shapes,
        )

        from executorch.backends.nvidia.tensorrt.converters import reshape  # noqa: F401

        self.assertTrue(supports_dynamic_shapes("aten.flatten.using_ints"))

    def test_select_supports_dynamic_shapes(self) -> None:
        from executorch.backends.nvidia.tensorrt.converter_registry import (
            supports_dynamic_shapes,
        )

        from executorch.backends.nvidia.tensorrt.converters import reshape  # noqa: F401

        self.assertTrue(supports_dynamic_shapes("aten.select.int"))
        self.assertTrue(supports_dynamic_shapes("aten.select_copy.int"))

    def test_non_dynamic_converter_returns_false(self) -> None:
        from executorch.backends.nvidia.tensorrt.converter_registry import (
            supports_dynamic_shapes,
        )

        from executorch.backends.nvidia.tensorrt.converters import add  # noqa: F401

        self.assertFalse(supports_dynamic_shapes("aten.add.Tensor"))


class TestExpandDynamicPartitioner(unittest.TestCase):
    """Tests that the partitioner accepts expand_copy with dynamic shapes."""

    def _get_edge_program(self, model, example_inputs, dynamic_shapes=None):
        """Helper to export and convert to edge program."""
        from executorch.exir import to_edge

        exported = export(model, example_inputs, dynamic_shapes=dynamic_shapes)
        return to_edge(exported).exported_program()

    def test_expand_static_accepted(self) -> None:
        """Static expand should always be accepted by the partitioner."""
        from executorch.backends.nvidia.tensorrt.partitioner.operator_support import (
            TensorRTOperatorSupport,
        )

        class ExpandModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: [2, 1, 4] -> expand to [2, 3, 4]
                return x.expand(2, 3, 4)

        model = ExpandModel()
        example_inputs = (torch.randn(2, 1, 4),)
        edge_program = self._get_edge_program(model, example_inputs)

        op_support = TensorRTOperatorSupport()
        found_expand = False

        for node in edge_program.graph_module.graph.nodes:
            if node.op == "call_function":
                op_name = op_support._get_op_name(node)
                target_name = op_support._remove_namespace(op_name)
                if "expand" in target_name:
                    found_expand = True
                    self.assertTrue(
                        op_support.is_node_supported({}, node),
                        f"Static expand node {node.name} should be supported",
                    )

        self.assertTrue(found_expand, "Should find at least one expand node")

    def test_expand_dynamic_accepted(self) -> None:
        """Expand with dynamic shapes should be accepted (converter supports it)."""
        from executorch.backends.nvidia.tensorrt.partitioner.operator_support import (
            TensorRTOperatorSupport,
        )

        # Import converters to register them with supports_dynamic_shapes=True
        from executorch.backends.nvidia.tensorrt.converters import expand  # noqa: F401

        class ExpandModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: [B, 1, 4] -> expand to [B, 3, 4]
                B = x.shape[0]
                return x.expand(B, 3, 4)

        model = ExpandModel()
        example_inputs = (torch.randn(2, 1, 4),)
        batch = Dim("batch", min=1, max=32)
        dynamic_shapes = {"x": {0: batch}}
        edge_program = self._get_edge_program(
            model, example_inputs, dynamic_shapes=dynamic_shapes
        )

        op_support = TensorRTOperatorSupport()
        found_expand = False

        for node in edge_program.graph_module.graph.nodes:
            if node.op == "call_function":
                op_name = op_support._get_op_name(node)
                target_name = op_support._remove_namespace(op_name)
                if "expand" in target_name:
                    found_expand = True
                    self.assertTrue(
                        op_support.is_node_supported({}, node),
                        f"Dynamic expand node {node.name} should be supported "
                        f"(converter declares supports_dynamic_shapes=True)",
                    )

        self.assertTrue(found_expand, "Should find at least one expand node")

    def test_expand_broadcast_mask_dynamic(self) -> None:
        """Typical attention mask broadcast: [B, 1, 1, S] -> [B, H, S, S]."""
        from executorch.backends.nvidia.tensorrt.partitioner.operator_support import (
            TensorRTOperatorSupport,
        )

        from executorch.backends.nvidia.tensorrt.converters import expand  # noqa: F401

        class MaskBroadcast(torch.nn.Module):
            def __init__(self, num_heads: int = 4):
                super().__init__()
                self.num_heads = num_heads

            def forward(self, mask: torch.Tensor) -> torch.Tensor:
                # mask: [B, 1, 1, S] -> [B, num_heads, S, S]
                B, _, _, S = mask.shape
                return mask.expand(B, self.num_heads, S, S)

        model = MaskBroadcast(num_heads=4)
        example_inputs = (torch.randn(2, 1, 1, 16),)
        batch = Dim("batch", min=1, max=32)
        seq = Dim("seq", min=1, max=512)
        dynamic_shapes = {"mask": {0: batch, 3: seq}}
        edge_program = self._get_edge_program(
            model, example_inputs, dynamic_shapes=dynamic_shapes
        )

        op_support = TensorRTOperatorSupport()
        found_expand = False

        for node in edge_program.graph_module.graph.nodes:
            if node.op == "call_function":
                op_name = op_support._get_op_name(node)
                target_name = op_support._remove_namespace(op_name)
                if "expand" in target_name:
                    found_expand = True
                    self.assertTrue(
                        op_support.is_node_supported({}, node),
                        f"Mask broadcast expand node {node.name} should be supported",
                    )

        self.assertTrue(found_expand, "Should find at least one expand node")


class TestReshapeDynamicPartitioner(unittest.TestCase):
    """Tests that reshape converters accept nodes with dynamic shapes."""

    def _get_edge_program(self, model, example_inputs, dynamic_shapes=None):
        """Helper to export and convert to edge program."""
        from executorch.exir import to_edge

        exported = export(model, example_inputs, dynamic_shapes=dynamic_shapes)
        return to_edge(exported).exported_program()

    def _assert_op_supported(self, edge_program, op_substr):
        """Assert that at least one node matching op_substr is supported."""
        from executorch.backends.nvidia.tensorrt.partitioner.operator_support import (
            TensorRTOperatorSupport,
        )

        op_support = TensorRTOperatorSupport()
        found = False

        for node in edge_program.graph_module.graph.nodes:
            if node.op == "call_function":
                op_name = op_support._get_op_name(node)
                target_name = op_support._remove_namespace(op_name)
                if op_substr in target_name:
                    found = True
                    self.assertTrue(
                        op_support.is_node_supported({}, node),
                        f"Dynamic {op_substr} node {node.name} should be supported",
                    )

        self.assertTrue(found, f"Should find at least one {op_substr} node")

    def test_unsqueeze_dynamic_accepted(self) -> None:
        """Unsqueeze with dynamic batch dim should be accepted."""
        from executorch.backends.nvidia.tensorrt.converters import reshape  # noqa: F401

        class UnsqueezeModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.unsqueeze(1)  # [B, D] -> [B, 1, D]

        model = UnsqueezeModel()
        example_inputs = (torch.randn(2, 16),)
        batch = Dim("batch", min=1, max=32)
        dynamic_shapes = {"x": {0: batch}}
        edge_program = self._get_edge_program(
            model, example_inputs, dynamic_shapes=dynamic_shapes
        )
        self._assert_op_supported(edge_program, "unsqueeze")

    def test_squeeze_dynamic_accepted(self) -> None:
        """Squeeze with dynamic batch dim should be accepted."""
        from executorch.backends.nvidia.tensorrt.converters import reshape  # noqa: F401

        class SqueezeModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.squeeze(1)  # [B, 1, D] -> [B, D]

        model = SqueezeModel()
        example_inputs = (torch.randn(2, 1, 16),)
        batch = Dim("batch", min=1, max=32)
        dynamic_shapes = {"x": {0: batch}}
        edge_program = self._get_edge_program(
            model, example_inputs, dynamic_shapes=dynamic_shapes
        )
        self._assert_op_supported(edge_program, "squeeze")

    def test_flatten_dynamic_accepted(self) -> None:
        """Flatten registry flag is set (to_edge decomposes flatten to view_copy)."""
        from executorch.backends.nvidia.tensorrt.converter_registry import (
            supports_dynamic_shapes,
        )

        from executorch.backends.nvidia.tensorrt.converters import reshape  # noqa: F401

        # to_edge decomposes flatten into view_copy, so we verify the
        # registry flag directly instead of testing via the partitioner.
        self.assertTrue(supports_dynamic_shapes("aten.flatten.using_ints"))

    def test_select_dynamic_accepted(self) -> None:
        """Select with dynamic batch dim should be accepted."""
        from executorch.backends.nvidia.tensorrt.converters import reshape  # noqa: F401

        class SelectModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.select(1, 0)  # [B, C, D] -> [B, D]

        model = SelectModel()
        example_inputs = (torch.randn(2, 4, 8),)
        batch = Dim("batch", min=1, max=32)
        dynamic_shapes = {"x": {0: batch}}
        edge_program = self._get_edge_program(
            model, example_inputs, dynamic_shapes=dynamic_shapes
        )
        self._assert_op_supported(edge_program, "select")


if __name__ == "__main__":
    unittest.main()
