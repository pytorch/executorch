# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.test.harness.tolerance import (
    _dtype_to_key,
    BACKEND_TOLERANCES,
    get_tolerance,
    ToleranceConfig,
)


class TestToleranceConfig(unittest.TestCase):
    def test_default_values(self):
        config = ToleranceConfig()
        self.assertEqual(config.atol, 1e-3)
        self.assertEqual(config.rtol, 1e-3)
        self.assertEqual(config.qtol, 0)

    def test_custom_values(self):
        config = ToleranceConfig(atol=1e-2, rtol=5e-2, qtol=2)
        self.assertEqual(config.atol, 1e-2)
        self.assertEqual(config.rtol, 5e-2)
        self.assertEqual(config.qtol, 2)

    def test_frozen(self):
        config = ToleranceConfig()
        with self.assertRaises(AttributeError):
            config.atol = 0.5  # type: ignore[misc]

    def test_with_quantization_scale(self):
        config = ToleranceConfig(atol=1e-3, rtol=1e-3, qtol=2)
        scale = 0.05
        adjusted = config.with_quantization_scale(scale)

        self.assertAlmostEqual(adjusted.atol, 1e-3 + 0.05 * 2)
        self.assertEqual(adjusted.rtol, 1e-3)
        self.assertEqual(adjusted.qtol, 2)

    def test_with_quantization_scale_zero_qtol(self):
        config = ToleranceConfig(atol=1e-3, rtol=1e-3, qtol=0)
        adjusted = config.with_quantization_scale(0.05)

        self.assertEqual(adjusted.atol, 1e-3)

    def test_with_quantization_scale_returns_new_instance(self):
        config = ToleranceConfig(atol=1e-3, rtol=1e-3, qtol=1)
        adjusted = config.with_quantization_scale(0.1)

        self.assertIsNot(config, adjusted)
        self.assertEqual(config.atol, 1e-3)

    def test_equality(self):
        a = ToleranceConfig(atol=1e-3, rtol=1e-3, qtol=0)
        b = ToleranceConfig(atol=1e-3, rtol=1e-3, qtol=0)
        self.assertEqual(a, b)

    def test_inequality(self):
        a = ToleranceConfig(atol=1e-3, rtol=1e-3, qtol=0)
        b = ToleranceConfig(atol=1e-2, rtol=1e-3, qtol=0)
        self.assertNotEqual(a, b)


class TestDtypeToKey(unittest.TestCase):
    def test_quantized_always_wins(self):
        self.assertEqual(_dtype_to_key(torch.float32, quantized=True), "quantized")
        self.assertEqual(_dtype_to_key(torch.float16, quantized=True), "quantized")
        self.assertEqual(_dtype_to_key(torch.bfloat16, quantized=True), "quantized")

    def test_fp16(self):
        self.assertEqual(_dtype_to_key(torch.float16, quantized=False), "fp16")

    def test_bf16(self):
        self.assertEqual(_dtype_to_key(torch.bfloat16, quantized=False), "bf16")

    def test_fp32_default(self):
        self.assertEqual(_dtype_to_key(torch.float32, quantized=False), "default")

    def test_fp64_default(self):
        self.assertEqual(_dtype_to_key(torch.float64, quantized=False), "default")

    def test_int_types_default(self):
        self.assertEqual(_dtype_to_key(torch.int32, quantized=False), "default")
        self.assertEqual(_dtype_to_key(torch.int64, quantized=False), "default")


class TestGetTolerance(unittest.TestCase):
    # --- Known backends ---

    def test_xnnpack_default(self):
        config = get_tolerance("xnnpack")
        self.assertEqual(config.atol, 1e-3)
        self.assertEqual(config.rtol, 1e-3)

    def test_xnnpack_fp16(self):
        config = get_tolerance("xnnpack", dtype=torch.float16)
        self.assertEqual(config.atol, 2e-3)
        self.assertEqual(config.rtol, 1e-3)

    def test_xnnpack_quantized(self):
        config = get_tolerance("xnnpack", quantized=True)
        self.assertEqual(config.atol, 1e-3)
        self.assertEqual(config.rtol, 1e-3)
        self.assertEqual(config.qtol, 1)

    def test_vulkan_default(self):
        config = get_tolerance("vulkan")
        self.assertEqual(config.atol, 1e-3)
        self.assertEqual(config.rtol, 1e-1)

    def test_vulkan_fp16(self):
        config = get_tolerance("vulkan", dtype=torch.float16)
        self.assertEqual(config.atol, 1e-2)
        self.assertEqual(config.rtol, 1e-2)

    def test_coreml_quantized(self):
        config = get_tolerance("coreml", quantized=True)
        self.assertEqual(config.atol, 5e-2)
        self.assertEqual(config.rtol, 5e-2)

    def test_metal_bf16(self):
        config = get_tolerance("metal", dtype=torch.bfloat16)
        self.assertEqual(config.atol, 1e-2)
        self.assertEqual(config.rtol, 1e-2)

    def test_metal_fp32(self):
        config = get_tolerance("metal")
        self.assertEqual(config.atol, 1e-5)
        self.assertEqual(config.rtol, 1e-5)

    def test_qnn_quantized(self):
        config = get_tolerance("qnn", quantized=True)
        self.assertEqual(config.atol, 1e-1)
        self.assertEqual(config.rtol, 1.0)

    def test_arm_quantized(self):
        config = get_tolerance("arm", quantized=True)
        self.assertEqual(config.qtol, 1)

    def test_arm_bf16(self):
        config = get_tolerance("arm", dtype=torch.bfloat16)
        self.assertEqual(config.atol, 1e-2)

    def test_nxp_default(self):
        config = get_tolerance("nxp")
        self.assertEqual(config.atol, 1e-8)
        self.assertEqual(config.rtol, 1e-5)

    def test_nxp_quantized(self):
        config = get_tolerance("nxp", quantized=True)
        self.assertEqual(config.atol, 1.0)

    def test_mlx_fp16(self):
        config = get_tolerance("mlx", dtype=torch.float16)
        self.assertEqual(config.atol, 1e-3)
        self.assertEqual(config.rtol, 1e-3)

    def test_mlx_bf16(self):
        config = get_tolerance("mlx", dtype=torch.bfloat16)
        self.assertEqual(config.atol, 1e-2)
        self.assertEqual(config.rtol, 1e-2)

    # --- Fallback behavior ---

    def test_unknown_backend_returns_global_default(self):
        config = get_tolerance("nonexistent_backend")
        self.assertEqual(config.atol, 1e-3)
        self.assertEqual(config.rtol, 1e-3)
        self.assertEqual(config.qtol, 0)

    def test_unknown_dtype_falls_back_to_backend_default(self):
        config = get_tolerance("xnnpack", dtype=torch.float64)
        self.assertEqual(config.atol, 1e-3)
        self.assertEqual(config.rtol, 1e-3)

    def test_samsung_no_quantized_falls_back_to_default(self):
        config = get_tolerance("samsung", quantized=True)
        self.assertEqual(config.atol, 1e-3)
        self.assertEqual(config.rtol, 1e-3)

    def test_webgpu_fp16_falls_back_to_default(self):
        config = get_tolerance("webgpu", dtype=torch.float16)
        self.assertEqual(config.atol, 1e-3)
        self.assertEqual(config.rtol, 1e-3)

    # --- Op parameter (reserved for future use) ---

    def test_op_parameter_currently_ignored(self):
        config_with_op = get_tolerance("xnnpack", op="aten.exp")
        config_without_op = get_tolerance("xnnpack")
        self.assertEqual(config_with_op, config_without_op)

    # --- Registry completeness ---

    def test_all_backends_have_default_key(self):
        for backend_name, backend_config in BACKEND_TOLERANCES.items():
            self.assertIn(
                "default",
                backend_config,
                f"Backend '{backend_name}' is missing a 'default' tolerance entry",
            )

    def test_all_tolerance_values_are_non_negative(self):
        for backend_name, backend_config in BACKEND_TOLERANCES.items():
            for dtype_key, config in backend_config.items():
                self.assertGreaterEqual(
                    config.atol,
                    0,
                    f"{backend_name}/{dtype_key} has negative atol",
                )
                self.assertGreaterEqual(
                    config.rtol,
                    0,
                    f"{backend_name}/{dtype_key} has negative rtol",
                )
                self.assertGreaterEqual(
                    config.qtol,
                    0,
                    f"{backend_name}/{dtype_key} has negative qtol",
                )

    def test_registry_has_expected_backends(self):
        expected = {
            "xnnpack",
            "vulkan",
            "coreml",
            "mps",
            "metal",
            "qnn",
            "arm",
            "cortex_m",
            "samsung",
            "mlx",
            "openvino",
            "nxp",
            "cadence",
            "cuda",
            "webgpu",
        }
        self.assertEqual(set(BACKEND_TOLERANCES.keys()), expected)
