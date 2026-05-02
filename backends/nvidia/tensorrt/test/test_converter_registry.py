# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for TensorRT converter registry and converter utilities."""

import unittest


class ConverterRegistryTest(unittest.TestCase):
    """Tests for converter registry functionality."""

    def test_registry_functions_exist(self) -> None:
        from executorch.backends.nvidia.tensorrt.converter_registry import (
            clear_registry,
            get_registered_ops,
            has_converter,
            lookup_converter,
            register_converter,
        )

        self.assertIsNotNone(has_converter)
        self.assertIsNotNone(lookup_converter)
        self.assertIsNotNone(register_converter)
        self.assertIsNotNone(get_registered_ops)
        self.assertIsNotNone(clear_registry)

    def test_add_converter_registered(self) -> None:
        from executorch.backends.nvidia.tensorrt.converter_registry import (
            get_registered_ops,
            has_converter,
            lookup_converter,
        )

        # Import converters to trigger registration via @converter decorator
        from executorch.backends.nvidia.tensorrt.converters import add  # noqa: F401

        self.assertTrue(has_converter("aten.add.Tensor"))
        self.assertIn("aten.add.Tensor", get_registered_ops())
        self.assertIsNotNone(lookup_converter("aten.add.Tensor"))

    def test_all_converters_registered(self) -> None:
        """Test that all converters are registered after importing converters."""
        from executorch.backends.nvidia.tensorrt.converter_registry import (
            get_registered_ops,
            has_converter,
        )

        # Import converters to trigger registration
        from executorch.backends.nvidia.tensorrt.converters import add  # noqa: F401

        expected_ops = [
            # Basic arithmetic
            "aten.add.Tensor",
        ]

        for op in expected_ops:
            self.assertTrue(has_converter(op), f"Missing converter for {op}")
            self.assertIn(op, get_registered_ops())


class ConverterUtilsTest(unittest.TestCase):
    """Tests for converter utility functions."""

    def test_converter_utils_functions_exist(self) -> None:
        from executorch.backends.nvidia.tensorrt.converter_utils import (
            broadcast_tensors,
            get_node_dtype,
            get_trt_tensor,
            set_layer_name,
            torch_dtype_to_trt,
            trt_dtype_to_torch,
        )

        self.assertIsNotNone(torch_dtype_to_trt)
        self.assertIsNotNone(trt_dtype_to_torch)
        self.assertIsNotNone(get_trt_tensor)
        self.assertIsNotNone(broadcast_tensors)
        self.assertIsNotNone(get_node_dtype)
        self.assertIsNotNone(set_layer_name)
