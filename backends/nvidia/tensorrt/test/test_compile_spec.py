# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for TensorRT compile spec."""

import unittest


class CompileSpecTest(unittest.TestCase):
    """Tests for TensorRTCompileSpec functionality."""

    def test_default_values(self) -> None:
        from executorch.backends.nvidia.tensorrt.compile_spec import (
            TensorRTCompileSpec,
            TensorRTPrecision,
        )

        spec = TensorRTCompileSpec()
        self.assertEqual(spec.workspace_size, 4 << 30)  # 4GB
        self.assertEqual(spec.precision, TensorRTPrecision.FP32)
        self.assertFalse(spec.strict_type_constraints)
        self.assertEqual(spec.max_batch_size, 1)
        self.assertEqual(spec.device_id, 0)
        self.assertEqual(spec.dla_core, -1)
        self.assertTrue(spec.allow_gpu_fallback)

    def test_custom_values(self) -> None:
        from executorch.backends.nvidia.tensorrt.compile_spec import (
            TensorRTCompileSpec,
            TensorRTPrecision,
        )

        spec = TensorRTCompileSpec(
            workspace_size=2 << 30,  # 2GB
            precision=TensorRTPrecision.FP16,
            strict_type_constraints=True,
            max_batch_size=8,
            device_id=1,
            dla_core=0,
            allow_gpu_fallback=False,
        )
        self.assertEqual(spec.workspace_size, 2 << 30)
        self.assertEqual(spec.precision, TensorRTPrecision.FP16)
        self.assertTrue(spec.strict_type_constraints)
        self.assertEqual(spec.max_batch_size, 8)
        self.assertEqual(spec.device_id, 1)
        self.assertEqual(spec.dla_core, 0)
        self.assertFalse(spec.allow_gpu_fallback)

    def test_serialization_roundtrip(self) -> None:
        from executorch.backends.nvidia.tensorrt.compile_spec import (
            TensorRTCompileSpec,
        )

        # Test with default values
        original = TensorRTCompileSpec()
        serialized = original.to_compile_specs()
        restored = TensorRTCompileSpec.from_compile_specs(serialized)

        self.assertIsNotNone(restored)
        self.assertEqual(original.workspace_size, restored.workspace_size)
        self.assertEqual(original.precision, restored.precision)

    def test_serialization_roundtrip_custom(self) -> None:
        from executorch.backends.nvidia.tensorrt.compile_spec import (
            TensorRTCompileSpec,
            TensorRTPrecision,
        )

        # Test with custom values
        original = TensorRTCompileSpec(
            workspace_size=512 << 20,  # 512MB
            precision=TensorRTPrecision.INT8,
            max_batch_size=16,
        )
        serialized = original.to_compile_specs()
        restored = TensorRTCompileSpec.from_compile_specs(serialized)

        self.assertIsNotNone(restored)
        self.assertEqual(original.workspace_size, restored.workspace_size)
        self.assertEqual(original.precision, restored.precision)
        self.assertEqual(original.max_batch_size, restored.max_batch_size)

    def test_from_empty_compile_specs(self) -> None:
        from executorch.backends.nvidia.tensorrt.compile_spec import (
            TensorRTCompileSpec,
        )

        result = TensorRTCompileSpec.from_compile_specs([])
        self.assertIsNone(result)

    def test_compile_spec_key(self) -> None:
        from executorch.backends.nvidia.tensorrt.compile_spec import (
            TENSORRT_COMPILE_SPEC_KEY,
            TensorRTCompileSpec,
        )

        spec = TensorRTCompileSpec()
        serialized = spec.to_compile_specs()

        self.assertEqual(len(serialized), 1)
        self.assertEqual(serialized[0].key, TENSORRT_COMPILE_SPEC_KEY)
