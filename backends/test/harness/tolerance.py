# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Centralized tolerance registry for ExecuTorch backend test harness.

Provides documented per-backend tolerance defaults extracted from existing tests,
a lookup API with fallback chain, and a ToleranceConfig dataclass for consistent
tolerance representation.

See https://github.com/pytorch/executorch/issues/19910 for context.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass(frozen=True)
class ToleranceConfig:
    """Tolerance configuration for numerical accuracy comparison.

    Attributes:
        atol: Absolute tolerance for torch.allclose comparison.
        rtol: Relative tolerance for torch.allclose comparison.
        qtol: Quantization tolerance in quantization steps. When a dequantization
              scale is detected, effective atol becomes atol + (scale * qtol).
    """

    atol: float = 1e-3
    rtol: float = 1e-3
    qtol: int = 0

    def with_quantization_scale(self, scale: float) -> "ToleranceConfig":
        """Return a new config with atol adjusted for the given quantization scale."""
        return ToleranceConfig(
            atol=self.atol + scale * self.qtol,
            rtol=self.rtol,
            qtol=self.qtol,
        )


# --- Backend tolerance registry ---
#
# Values extracted from a cross-backend audit of all ExecuTorch test files.
# Each backend's defaults reflect what is currently hardcoded across its tests.
#
# Dtype keys:
#   "default"   — float32 or unspecified dtype
#   "fp16"      — float16
#   "bf16"      — bfloat16
#   "quantized" — int8/uint8 quantized outputs
#
# When adding a new backend, document why the chosen tolerances are appropriate.

BACKEND_TOLERANCES: Dict[str, Dict[str, ToleranceConfig]] = {
    "xnnpack": {
        # FP32: 1e-3 is the harness default; ~90% of XNNPACK op tests use this.
        "default": ToleranceConfig(atol=1e-3, rtol=1e-3),
        # FP16: varies per op (exp/gelu use dynamic calc, cos uses 2e-3).
        # 2e-3 is a reasonable general default for fp16 ops.
        "fp16": ToleranceConfig(atol=2e-3, rtol=1e-3),
        # Quantized: most tests use qtol=1; conv2d uses qtol=2.
        "quantized": ToleranceConfig(atol=1e-3, rtol=1e-3, qtol=1),
    },
    "vulkan": {
        # Python delegate tests: atol=1e-3, rtol=1e-1.
        "default": ToleranceConfig(atol=1e-3, rtol=1e-1),
        # FP16: generated op tests and custom ops use 1e-2.
        "fp16": ToleranceConfig(atol=1e-2, rtol=1e-2),
        # Quantized: delegate tests use atol=1e-2, rtol=1e-1.
        "quantized": ToleranceConfig(atol=1e-2, rtol=1e-1),
    },
    "coreml": {
        # General E2E tests use 1e-2/1e-2.
        "default": ToleranceConfig(atol=1e-2, rtol=1e-2),
        # Multifunction/KV-cache tests use tighter 1e-4.
        "fp16": ToleranceConfig(atol=1e-2, rtol=1e-2),
        # Quantizer tests use 5e-2 via np.testing.assert_allclose.
        "quantized": ToleranceConfig(atol=5e-2, rtol=5e-2),
    },
    "mps": {
        "default": ToleranceConfig(atol=1e-3, rtol=1e-3),
        # FP16 linear tests use atol=5e-2.
        "fp16": ToleranceConfig(atol=5e-2, rtol=1e-3),
    },
    "metal": {
        # FP32 default from DEFAULT_TOLERANCES in test_modules.py.
        "default": ToleranceConfig(atol=1e-5, rtol=1e-5),
        # BF16 from DEFAULT_TOLERANCES.
        "bf16": ToleranceConfig(atol=1e-2, rtol=1e-2),
    },
    "qnn": {
        # All FP16 HTP/GPU tests: atol=1e-1, rtol=1e-1.
        "default": ToleranceConfig(atol=1e-1, rtol=1e-1),
        "fp16": ToleranceConfig(atol=1e-1, rtol=1e-1),
        # Quantized: atol=1e-1, rtol=1.0. The high rtol reflects
        # Qualcomm HTP fixed-point arithmetic variance.
        "quantized": ToleranceConfig(atol=1e-1, rtol=1.0),
    },
    "arm": {
        # Shared harness defaults for TOSA pipeline.
        "default": ToleranceConfig(atol=1e-3, rtol=1e-3),
        "bf16": ToleranceConfig(atol=1e-2, rtol=1e-2),
        # INT pipeline default: qtol=1.
        "quantized": ToleranceConfig(atol=1e-3, rtol=1e-3, qtol=1),
    },
    "cortex_m": {
        "default": ToleranceConfig(atol=1e-3, rtol=1e-3),
        # Most quantized op tests use qtol=1; conv uses qtol=2.
        "quantized": ToleranceConfig(atol=1e-3, rtol=1e-3, qtol=1),
    },
    "samsung": {
        "default": ToleranceConfig(atol=1e-3, rtol=1e-3),
    },
    "mlx": {
        # From MLX's TOLERANCE_PRESETS.
        "default": ToleranceConfig(atol=1e-5, rtol=1e-5),
        "fp16": ToleranceConfig(atol=1e-3, rtol=1e-3),
        "bf16": ToleranceConfig(atol=1e-2, rtol=1e-2),
        "quantized": ToleranceConfig(atol=1e-1, rtol=1e-1),
    },
    "openvino": {
        "default": ToleranceConfig(atol=1e-3, rtol=1e-3),
    },
    "nxp": {
        # NXP uses numpy defaults: rtol=1e-5, atol=1e-8.
        "default": ToleranceConfig(atol=1e-8, rtol=1e-5),
        # Quantized ops typically use atol=1.0 (one int8 step).
        "quantized": ToleranceConfig(atol=1.0, rtol=1e-5),
    },
    "cadence": {
        # Cadence uses RMS-based comparison; these are approximate equivalents
        # for the pass test helpers that do use torch.allclose.
        "default": ToleranceConfig(atol=1e-6, rtol=1e-5),
        "quantized": ToleranceConfig(atol=5.0, rtol=0.05),
    },
    "cuda": {
        # SDPA tests: MAX_ABS_TOL = 1e-2.
        "default": ToleranceConfig(atol=1e-2, rtol=1e-2),
        "bf16": ToleranceConfig(atol=1e-2, rtol=1e-2),
    },
    "webgpu": {
        # Native C++ test: max error threshold 1e-3.
        "default": ToleranceConfig(atol=1e-3, rtol=1e-3),
    },
}

# Global fallback when backend is unknown.
_GLOBAL_DEFAULT = ToleranceConfig(atol=1e-3, rtol=1e-3, qtol=0)


def _dtype_to_key(dtype: torch.dtype, quantized: bool) -> str:
    """Map a torch dtype and quantization flag to a registry lookup key."""
    if quantized:
        return "quantized"
    dtype_key_map = {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }
    return dtype_key_map.get(dtype, "default")


def get_tolerance(
    backend: str,
    dtype: torch.dtype = torch.float32,
    quantized: bool = False,
    op: Optional[str] = None,
) -> ToleranceConfig:
    """Look up tolerance config for a backend/dtype/op combination.

    Lookup order (first match wins):
        1. Backend + dtype-specific (e.g. xnnpack + fp16)
        2. Backend default
        3. Global default (atol=1e-3, rtol=1e-3, qtol=0)

    Args:
        backend: Backend identifier (e.g. "xnnpack", "vulkan", "arm").
        dtype: Output tensor dtype. Used to select fp16/bf16 tolerance tiers.
        quantized: Whether the output is from a quantized model.
        op: Reserved for future per-op overrides. Currently unused.

    Returns:
        ToleranceConfig with appropriate atol, rtol, and qtol values.
    """
    backend_config = BACKEND_TOLERANCES.get(backend, {})

    dtype_key = _dtype_to_key(dtype, quantized)
    if dtype_key in backend_config:
        return backend_config[dtype_key]

    if "default" in backend_config:
        return backend_config["default"]

    return _GLOBAL_DEFAULT
