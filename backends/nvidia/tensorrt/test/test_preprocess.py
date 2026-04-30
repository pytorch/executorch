# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for TensorRT preprocess and backend functionality."""

import unittest

import torch

# Check TensorRT availability at module level for skip decorators
_TENSORRT_AVAILABLE = False
try:
    import tensorrt  # noqa: F401

    _TENSORRT_AVAILABLE = True
except ImportError:
    pass


def _requires_tensorrt(test_class):
    """Class decorator to skip all tests in a class if TensorRT is not available."""
    if not _TENSORRT_AVAILABLE:
        return unittest.skip("TensorRT is not available")(test_class)
    return test_class


class _AddModel(torch.nn.Module):
    """Simple add model for testing."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class TensorRTBackendTest(unittest.TestCase):
    """Tests for TensorRTBackend class structure."""

    @unittest.skipUnless(_TENSORRT_AVAILABLE, "TensorRT is not available")
    def test_backend_import(self) -> None:
        """Test that TensorRTBackend can be imported."""
        from executorch.backends.nvidia.tensorrt.backend import TensorRTBackend

        self.assertIsNotNone(TensorRTBackend)
        self.assertTrue(hasattr(TensorRTBackend, "preprocess"))

    @unittest.skipUnless(_TENSORRT_AVAILABLE, "TensorRT is not available")
    def test_preprocess_signature(self) -> None:
        """Test that preprocess has the correct signature."""
        import inspect

        from executorch.backends.nvidia.tensorrt.backend import TensorRTBackend

        sig = inspect.signature(TensorRTBackend.preprocess)
        params = list(sig.parameters.keys())
        self.assertIn("edge_program", params)
        self.assertIn("compile_specs", params)


@_requires_tensorrt
class TensorRTPreprocessTest(unittest.TestCase):
    """Integration tests for TensorRT preprocess (requires TensorRT)."""

    def test_preprocess_add_model(self) -> None:
        """Test that preprocess produces non-empty bytes for add model."""
        from executorch.backends.nvidia.tensorrt.compile_spec import TensorRTCompileSpec
        from executorch.backends.nvidia.tensorrt.backend import TensorRTBackend

        model = _AddModel()
        model.eval()

        example_inputs = (torch.randn(2, 3), torch.randn(2, 3))
        exported = torch.export.export(model, example_inputs)

        compile_spec = TensorRTCompileSpec()
        result = TensorRTBackend.preprocess(exported, compile_spec.to_compile_specs())

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.processed_bytes)
        self.assertGreater(len(result.processed_bytes), 0)

    def test_preprocess_empty_compile_specs(self) -> None:
        """Test that preprocess works with empty compile specs (uses defaults)."""
        from executorch.backends.nvidia.tensorrt.backend import TensorRTBackend

        model = _AddModel()
        model.eval()

        example_inputs = (torch.randn(2, 3), torch.randn(2, 3))
        exported = torch.export.export(model, example_inputs)

        result = TensorRTBackend.preprocess(exported, [])

        self.assertIsNotNone(result)
        self.assertGreater(len(result.processed_bytes), 0)

    def test_preprocess_different_shapes(self) -> None:
        """Test preprocess with different input shapes."""
        from executorch.backends.nvidia.tensorrt.compile_spec import TensorRTCompileSpec
        from executorch.backends.nvidia.tensorrt.backend import TensorRTBackend

        model = _AddModel()
        model.eval()

        example_inputs = (torch.randn(4, 8, 16), torch.randn(4, 8, 16))
        exported = torch.export.export(model, example_inputs)

        compile_spec = TensorRTCompileSpec()
        result = TensorRTBackend.preprocess(exported, compile_spec.to_compile_specs())

        self.assertIsNotNone(result)
        self.assertGreater(len(result.processed_bytes), 0)
