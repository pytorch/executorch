# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for Metal backend modules.

These tests export and run various model modules through the Metal backend
to verify that the export and execution pipeline works correctly.

These tests require MPS to be available. On systems without MPS support,
the export tests will be skipped.
"""

import os
import platform
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from executorch.backends.apple.metal.metal_backend import MetalBackend
from executorch.backends.apple.metal.metal_partitioner import MetalPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch import nn
from torch.export import export
from torch.nn.attention import SDPBackend

try:
    # Need to import to load the ops
    import torchao.experimental.ops.mps  # noqa: F401
    from torchao.experimental.quant_api import UIntxWeightOnlyConfig
    from torchao.quantization.quant_api import quantize_

    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False


# Check if MPS is available for export tests
MPS_AVAILABLE = torch.backends.mps.is_available()
IS_MACOS = platform.system() == "Darwin"
SKIP_EXPORT_TESTS = not MPS_AVAILABLE
SKIP_REASON = "MPS not available - Metal export tests require MPS support"

# Check if running in CI (GitHub Actions)
IS_CI = os.environ.get("GITHUB_ACTIONS") == "true"

# Paths
TESTS_DIR = Path(__file__).parent
EXECUTORCH_ROOT = TESTS_DIR.parent.parent.parent.parent
BUILD_DIR = EXECUTORCH_ROOT / "cmake-out"
EXECUTOR_RUNNER = BUILD_DIR / "executor_runner"
RUN_METAL_TEST_SCRIPT = TESTS_DIR / "run_metal_test.sh"

# Test output directory - use current working directory in CI for reliable write access
if IS_CI:
    TEST_OUTPUT_BASE_DIR = Path.cwd() / "metal_backend_module_outputs"
else:
    TEST_OUTPUT_BASE_DIR = None  # Will use tempfile.TemporaryDirectory

# Check if executor_runner is built
EXECUTOR_RUNNER_AVAILABLE = EXECUTOR_RUNNER.exists()
SKIP_RUNTIME_TESTS = not EXECUTOR_RUNNER_AVAILABLE or SKIP_EXPORT_TESTS
SKIP_RUNTIME_REASON = (
    "executor_runner not built - run 'backends/apple/metal/tests/run_metal_test.sh --build'"
    if not EXECUTOR_RUNNER_AVAILABLE
    else SKIP_REASON
)

# Data types to test
DTYPES = [torch.float32, torch.bfloat16]

# Map dtype to short name for test method naming
DTYPE_NAMES = {
    torch.float32: "float32",
    torch.bfloat16: "bfloat16",
}

# Default tolerances for output comparison by dtype
# bfloat16 has lower precision (7 bits mantissa vs 23 for float32)
DEFAULT_TOLERANCES = {
    torch.float32: {"atol": 1e-5, "rtol": 1e-5},
    torch.bfloat16: {"atol": 1e-2, "rtol": 1e-2},
}


# Registry mapping model names to their configurations
# Each entry can optionally include:
#   - "atol": float - Override absolute tolerance for all dtypes
#   - "rtol": float - Override relative tolerance for all dtypes
#   - "atol_<dtype>": float - Override absolute tolerance for specific dtype (e.g., "atol_bfloat16")
#   - "rtol_<dtype>": float - Override relative tolerance for specific dtype (e.g., "rtol_bfloat16")
#   - "skip": bool or str - Skip all tests for this module (True to skip, or string with reason)
#   - "skip_<dtype>": bool or str - Skip tests for specific dtype (e.g., "skip_bfloat16")
#   - "qlinear": str - Quantization config for linear layers (e.g., "fpa4w" for 4-bit weights)
#   - "qlinear_group_size": int - Group size for quantization (default: 32)
#   - "compare_to_unquantized": bool - If True, compare quantized model output to unquantized reference (default: True for quantized models)
#
# Quantization Usage:
#   To enable int4 quantization for a module, add "qlinear": "fpa4w" to its registry entry.
#   This applies 4-bit weight quantization (floating point activation, 4-bit weight) using torchao.
#   The quantization is applied after converting the model to the specified dtype but before export.
#
#   By default, quantized models are compared against unquantized reference models to measure
#   the actual quantization error. Set "compare_to_unquantized": False to compare against
#   the quantized PyTorch model instead.
#
#   Example:
#     MODULE_REGISTRY["my_linear_model"] = {
#         "model_class": MyLinearModel,
#         "input_shapes": [(128, 256)],
#         "description": "My linear model with int4 quantization",
#         "qlinear": "fpa4w",
#         "qlinear_group_size": 32,
#         "compare_to_unquantized": True,  # Compare to unquantized reference
#         "atol_float32": 5e-2,  # Quantization reduces precision, so increase tolerance
#         "rtol_float32": 5e-2,
#     }
#
# Model Parameter Initialization:
#   Model parameters are initialized with their default dtype (typically float32) when the
#   model class is instantiated. The parameters are then converted to the target dtype using
#   model.to(dtype). For example:
#     - nn.Parameter(torch.arange(20, dtype=torch.get_default_dtype()) creates float32 parameters
#     - These are converted to bfloat16 when model.to(torch.bfloat16) is called


MODULE_REGISTRY: Dict[str, Dict[str, Any]] = {}


# =============================================================================
# Model Definitions
# =============================================================================


class Add(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return x + y


MODULE_REGISTRY["add"] = {
    "model_class": Add,
    "input_shapes": [(10,), (10,)],
    "description": "Simple tensor addition model",
}


# -------------------------------------------------------------------------
# Matrix Multiplication Modules
# -------------------------------------------------------------------------


class Mm(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return x.mm(y)


MODULE_REGISTRY["mm"] = {
    "model_class": Mm,
    "input_shapes": [(3, 4), (4, 5)],
    "description": "Simple mm layer model",
}


# -------------------------------------------------------------------------
class MmWeightParam(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.arange(20, dtype=torch.get_default_dtype()).reshape(4, 5)
        )

    def forward(self, x: torch.Tensor):
        return x.mm(self.weight)


MODULE_REGISTRY["mm_weight_param"] = {
    "model_class": MmWeightParam,
    "input_shapes": [(3, 4)],
    "description": "Matrix multiplication with weight parameter",
}


# -------------------------------------------------------------------------
# Linear Modules
# -------------------------------------------------------------------------


class LinearNoBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(7, 101, bias=False)

    def forward(self, x: torch.Tensor):
        return self.linear(x)


MODULE_REGISTRY["linear_nobias"] = {
    "model_class": LinearNoBias,
    "input_shapes": [(127, 7)],
    "description": "Simple linear layer model with no bias",
}


# -------------------------------------------------------------------------
class LinearWithBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(7, 101, bias=True)

    def forward(self, x: torch.Tensor):
        return self.linear(x)


MODULE_REGISTRY["linear_bias"] = {
    "model_class": LinearWithBias,
    "input_shapes": [(127, 7)],
    "description": "Simple linear layer model bias",
}


# -------------------------------------------------------------------------
class LinearNoBiasInt4(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 256, bias=False)

    def forward(self, x: torch.Tensor):
        return self.linear(x)


MODULE_REGISTRY["linear_nobias_int4"] = {
    "model_class": LinearNoBiasInt4,
    "input_shapes": [(127, 128)],
    "description": "Linear layer without bias and int4 quantization",
    "qlinear": "fpa4w",
    "qlinear_group_size": 32,
    "compare_to_unquantized": False,
    "atol_float32": 5e-2,
    "rtol_float32": 5e-2,
    "atol_bfloat16": 1e-1,
    "rtol_bfloat16": 1e-1,
    "skip": not TORCHAO_AVAILABLE,
}


# -------------------------------------------------------------------------
class LinearWithBiasInt4(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 256, bias=True)

    def forward(self, x: torch.Tensor):
        return self.linear(x)


MODULE_REGISTRY["linear_bias_int4"] = {
    "model_class": LinearWithBiasInt4,
    "input_shapes": [(127, 128)],
    "description": "Linear layer with bias and int4 quantization",
    "qlinear": "fpa4w",
    "qlinear_group_size": 32,
    "compare_to_unquantized": False,
    "atol_float32": 5e-2,
    "rtol_float32": 5e-2,
    "atol_bfloat16": 1e-1,
    "rtol_bfloat16": 1e-1,
    "skip": not TORCHAO_AVAILABLE,
}


# -------------------------------------------------------------------------
class LinearInt4_QMV_IMPL(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(640, 8198, bias=True)

    def forward(self, x: torch.Tensor):
        return self.linear(x)


MODULE_REGISTRY["linear_int4_qmv_impl"] = {
    "model_class": LinearInt4_QMV_IMPL,
    "input_shapes": [(1, 640)],
    "description": "Linear int4 quantization dispatching to qmv_impl",
    "qlinear": "fpa4w",
    "qlinear_group_size": 32,
    "compare_to_unquantized": False,
    "atol_float32": 5e-2,
    "rtol_float32": 5e-2,
    "atol_bfloat16": 1e-1,
    "rtol_bfloat16": 1e-1,
    "skip": not TORCHAO_AVAILABLE,
}


# -------------------------------------------------------------------------
# Convolution Modules
# -------------------------------------------------------------------------


class Conv1dNoBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=8,
            out_channels=6,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=False,
        )

    def forward(self, x):
        return self.conv(x)


MODULE_REGISTRY["conv1d_nobias"] = {
    "model_class": Conv1dNoBias,
    "input_shapes": [(1, 8, 5)],
    "description": "Conv1d layer with 8 input channels, 6 output channels",
}


# -------------------------------------------------------------------------
class Conv1dBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=80,
            out_channels=384,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=True,
        )

    def forward(self, x):
        return self.conv(x)


MODULE_REGISTRY["conv1d_bias"] = {
    "model_class": Conv1dBias,
    "input_shapes": [(1, 80, 3000)],
    "description": "Conv1d layer with 80 input channels, 384 output channels",
}


# -------------------------------------------------------------------------
class Conv1dVoxtral(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=128,
            out_channels=1280,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=False,
        )

    def forward(self, x):
        return self.conv(x)


MODULE_REGISTRY["conv1d_voxtral"] = {
    "model_class": Conv1dVoxtral,
    "input_shapes": [(10, 128, 3000)],
    "description": "Conv1d layer with 128 input channels, 1280 output channels",
}


# -------------------------------------------------------------------------
# Attention (SDPA) Modules
# -------------------------------------------------------------------------


class SDPA(nn.Module):
    """Minimal SDPA test model."""

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        return output


MODULE_REGISTRY["sdpa"] = {
    "model_class": SDPA,
    "input_shapes": [(2, 4, 16, 64), (2, 4, 16, 64), (2, 4, 16, 64)],
    "description": "Simple Scaled Dot Product Attention model",
}


# -------------------------------------------------------------------------
class SDPAAdd(nn.Module):
    """SDPA model with Q, K, V as parameters that adds input to SDPA output."""

    def __init__(self, batch_size=2, num_heads=4, seq_len=16, head_dim=64):
        super().__init__()
        self.query = nn.Parameter(torch.randn(batch_size, num_heads, seq_len, head_dim))
        self.key = nn.Parameter(torch.randn(batch_size, num_heads, seq_len, head_dim))
        self.value = nn.Parameter(torch.randn(batch_size, num_heads, seq_len, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sdpa_output = torch.nn.functional.scaled_dot_product_attention(
            self.query, self.key, self.value, dropout_p=0.0, is_causal=False
        )
        return sdpa_output + x


MODULE_REGISTRY["sdpa_add"] = {
    "model_class": SDPAAdd,
    "input_shapes": [(2, 4, 16, 64)],
    "description": "SDPA model with Q,K,V as parameters that adds input to output",
}


# -------------------------------------------------------------------------
class BaseStridedSDPA(nn.Module):
    """SDPA model with strided Q, K, V parameters."""

    def __init__(
        self, q_size, k_size, v_size, q_stride, k_stride, v_stride, attn_mask_size=None
    ):
        super().__init__()
        self.q_size = q_size
        self.k_size = k_size
        self.v_size = v_size
        self.q_stride = q_stride
        self.k_stride = k_stride
        self.v_stride = v_stride
        self.attn_mask_size = attn_mask_size

        self.query = nn.Parameter(torch.randn(q_size))
        self.key = nn.Parameter(torch.randn(k_size))
        self.value = nn.Parameter(torch.randn(v_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = torch.as_strided(self.query, size=self.q_size, stride=self.q_stride)
        key = torch.as_strided(self.key, size=self.k_size, stride=self.k_stride)
        value = torch.as_strided(self.value, size=self.v_size, stride=self.v_stride)
        attn_mask = None
        if self.attn_mask_size:
            attn_mask = torch.zeros(self.attn_mask_size)

        sdpa_output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask, dropout_p=0.0, is_causal=False, scale=1.0
        )
        return sdpa_output + x


# -------------------------------------------------------------------------
class SDPAStrided(BaseStridedSDPA):
    def __init__(self):
        super().__init__(
            q_size=(10, 20, 1500, 64),
            k_size=(10, 20, 1500, 64),
            v_size=(10, 20, 1500, 64),
            q_stride=(1920000, 64, 1280, 1),
            k_stride=(1920000, 64, 1280, 1),
            v_stride=(1920000, 64, 1280, 1),
        )


MODULE_REGISTRY["sdpa_strided"] = {
    "model_class": SDPAStrided,
    "input_shapes": [(10, 20, 1500, 64)],
    "description": "Audio Encoder model with strided SDPA",
    "atol_float32": 1e-4,
    "atol_bfloat16": 5e-2,
}


# -------------------------------------------------------------------------
class SDPAStridedBroadcast(BaseStridedSDPA):
    def __init__(self):
        super().__init__(
            q_size=(1, 20, 1, 64),
            k_size=(1, 20, 1500, 64),
            v_size=(1, 20, 1500, 64),
            q_stride=(1280, 64, 1280, 1),
            k_stride=(1920000, 64, 1280, 1),
            v_stride=(1920000, 64, 1280, 1),
        )


MODULE_REGISTRY["sdpa_strided_broadcast"] = {
    "model_class": SDPAStridedBroadcast,
    "input_shapes": [(1, 20, 1, 64)],
    "description": "Whisper-like strided SDPA variant 1",
}


# -------------------------------------------------------------------------
class SDPAStridedBroadcastAttnMask(BaseStridedSDPA):
    def __init__(self):
        super().__init__(
            q_size=(1, 20, 1, 64),
            k_size=(1, 20, 1024, 64),
            v_size=(1, 20, 1024, 64),
            q_stride=(1280, 64, 1280, 1),
            k_stride=(1310720, 65536, 64, 1),
            v_stride=(1310720, 65536, 64, 1),
            attn_mask_size=(1, 1, 1, 1024),
        )


MODULE_REGISTRY["sdpa_strided_broadcast_attn_mask"] = {
    "model_class": SDPAStridedBroadcastAttnMask,
    "input_shapes": [(1, 20, 1, 64)],
    "description": "Whisper-like strided SDPA variant 2",
}


# =============================================================================
# Helper Functions
# =============================================================================


def get_tolerances_for_model(
    model_name: str, dtype: torch.dtype
) -> Tuple[float, float]:
    """
    Get atol and rtol tolerances for a specific model and dtype.

    Priority order:
    1. Model-specific dtype tolerance (e.g., "atol_bfloat16")
    2. Model-specific general tolerance (e.g., "atol")
    3. Default dtype tolerance from DEFAULT_TOLERANCES

    Returns:
        Tuple of (atol, rtol)
    """
    model_config = MODULE_REGISTRY.get(model_name, {})
    dtype_name = DTYPE_NAMES.get(dtype, "float32")
    default_tols = DEFAULT_TOLERANCES.get(dtype, DEFAULT_TOLERANCES[torch.float32])

    # Check for dtype-specific override, then general override, then default
    atol = model_config.get(
        f"atol_{dtype_name}", model_config.get("atol", default_tols["atol"])
    )
    rtol = model_config.get(
        f"rtol_{dtype_name}", model_config.get("rtol", default_tols["rtol"])
    )

    return atol, rtol


def should_skip_model(model_name: str, dtype: torch.dtype) -> Tuple[bool, str]:
    """
    Check if a model should be skipped for testing.

    Priority order:
    1. Model-specific dtype skip (e.g., "skip_bfloat16")
    2. Model-specific general skip (e.g., "skip")

    Returns:
        Tuple of (should_skip, reason)
    """
    model_config = MODULE_REGISTRY.get(model_name, {})
    dtype_name = DTYPE_NAMES.get(dtype, "float32")

    # Check for dtype-specific skip first, then general skip
    skip_value = model_config.get(f"skip_{dtype_name}", model_config.get("skip", False))

    if skip_value is True:
        return True, f"{model_name} is marked as skipped"
    elif isinstance(skip_value, str):
        return True, skip_value
    return False, ""


def get_model_and_inputs(
    model_name: str,
    dtype: torch.dtype = torch.float32,
    qlinear: Optional[str] = None,
    qlinear_group_size: Optional[int] = None,
) -> Tuple[nn.Module, Tuple[torch.Tensor, ...]]:
    """Get model and example inputs based on model name.

    Note: Model parameters are initialized with their default dtype (typically float32)
    during model instantiation, then converted to the target dtype using model.to(dtype).

    Args:
        model_name: Name of the model to create
        dtype: Target data type for the model (default: torch.float32)
        qlinear: Optional quantization config (e.g., "fpa4w" for 4-bit weights).
                 If None, uses value from MODULE_REGISTRY if present.
        qlinear_group_size: Group size for quantization. If None, uses value from
                            MODULE_REGISTRY if present, otherwise defaults to 32.

    Returns:
        Tuple of (model, example_inputs)
    """
    if model_name not in MODULE_REGISTRY:
        available_models = ", ".join(MODULE_REGISTRY.keys())
        raise ValueError(
            f"Unsupported model: {model_name}. Available models: {available_models}"
        )

    model_config = MODULE_REGISTRY[model_name]
    model_class = model_config["model_class"]
    input_shapes = model_config["input_shapes"]

    # Use registry values if not explicitly provided
    if qlinear is None:
        qlinear = model_config.get("qlinear")
    if qlinear_group_size is None:
        qlinear_group_size = model_config.get("qlinear_group_size", 32)

    # Create model with default parameter dtypes (typically float32)
    model = model_class().eval()

    # Convert model parameters to target dtype if specified
    if dtype is not None:
        model = model.to(dtype)

    # Apply quantization if requested
    if qlinear is not None:
        quantize_model(model, qlinear, qlinear_group_size)

    example_inputs = tuple(torch.randn(*shape, dtype=dtype) for shape in input_shapes)

    return model, example_inputs


def quantize_model(model: nn.Module, qlinear: str, qlinear_group_size: int = 32):
    """Apply quantization to the model's linear layers.

    Args:
        model: The model to quantize (in-place).
        qlinear: Quantization config. Options:
            - "fpa4w": Floating point activation, 4-bit weight (Metal backend)
        qlinear_group_size: Group size for quantization (default: 32).
    """
    if not TORCHAO_AVAILABLE:
        raise RuntimeError(
            "torchao is not available. Install torchao to use quantization."
        )

    if qlinear == "fpa4w":
        linear_config = UIntxWeightOnlyConfig(
            group_size=qlinear_group_size,
            bitwidth=4,
        )
    else:
        raise ValueError(f"Unsupported linear quantization config '{qlinear}'.")

    def linear_filter(module, fqn):
        if isinstance(module, torch.nn.Linear):
            # Check if hidden dimension is divisible by group size
            return qlinear_group_size == 0 or (
                module.weight.shape[1] % qlinear_group_size == 0
            )
        return False

    quantize_(model, linear_config, filter_fn=linear_filter)


def export_model_to_metal(
    model: nn.Module, example_inputs: Tuple[torch.Tensor, ...]
) -> Any:
    """Export model through the Metal backend pipeline."""
    method_name = "forward"

    with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
        aten_dialect = export(model, example_inputs, strict=False)

        edge_program = to_edge_transform_and_lower(
            aten_dialect,
            partitioner=[
                MetalPartitioner(
                    [MetalBackend.generate_method_name_compile_spec(method_name)]
                )
            ],
        )

    executorch_program = edge_program.to_executorch()
    return executorch_program


def export_model_to_pte(
    model: nn.Module,
    example_inputs: Tuple[torch.Tensor, ...],
    output_dir: Path,
    model_name: str,
    compare_to_unquantized: bool = False,
    model_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, torch.Tensor]:
    """
    Export model to .pte file and compute expected output.

    Args:
        model: Model to export (may be quantized)
        example_inputs: Example inputs for export
        output_dir: Directory to save output files
        model_name: Name of the model
        compare_to_unquantized: If True and model has quantization config,
                                compute expected output from unquantized model
        model_config: Model configuration from MODULE_REGISTRY

    Returns:
        Tuple of (pte_path, expected_output)
    """
    # Compute expected output using all-ones input (matching export_aoti_metal.py)
    all_ones_input = tuple(torch.ones_like(inp) for inp in example_inputs)

    with torch.no_grad():
        if compare_to_unquantized and model_config and model_config.get("qlinear"):
            # Create unquantized reference model for comparison
            dtype = example_inputs[0].dtype if example_inputs else torch.float32
            model_class = model_config["model_class"]
            reference_model = model_class().eval()
            reference_model = reference_model.to(dtype)
            expected_output = reference_model(*all_ones_input)
        else:
            # Use the quantized model's output
            # For quantized models, torchao operators require MPS device
            if model_config and model_config.get("qlinear") and MPS_AVAILABLE:
                # Move model and inputs to MPS
                model_mps = model.to("mps")
                all_ones_input_mps = tuple(inp.to("mps") for inp in all_ones_input)
                expected_output = model_mps(*all_ones_input_mps)
                # Move output back to CPU for comparison
                expected_output = expected_output.cpu()
                # Move model back to CPU for export
                model = model_mps.to("cpu")
            else:
                # Non-quantized model, run on CPU
                expected_output = model(*all_ones_input)

    # Export to executorch
    executorch_program = export_model_to_metal(model, example_inputs)

    # Save .pte file (Metal backend embeds data into the .pte file, no separate .ptd)
    pte_path = output_dir / f"{model_name}.pte"
    with open(pte_path, "wb") as f:
        f.write(executorch_program.buffer)

    return pte_path, expected_output


def run_executor_runner(
    pte_path: Path, output_path: Path
) -> Tuple[bool, Optional[str]]:
    """
    Run the executor_runner binary with the given model files.

    Args:
        pte_path: Path to the .pte model file
        output_path: Base path for output files (executor_runner will create <output_path>-0.bin, etc.)

    Returns:
        Tuple of (success, error_message). If success is True, error_message is None.
        If success is False, error_message contains details about the failure.
    """
    if not EXECUTOR_RUNNER.exists():
        raise RuntimeError(
            f"executor_runner not found at {EXECUTOR_RUNNER}. "
            f"Run '{RUN_METAL_TEST_SCRIPT} --build' to build."
        )

    cmd = [
        str(EXECUTOR_RUNNER),
        "--model_path",
        str(pte_path),
        "--output_file",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(EXECUTORCH_ROOT),
        )
        if result.returncode == 0:
            return True, None
        else:
            error_msg = (
                f"executor_runner exited with code {result.returncode}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )
            return False, error_msg
    except subprocess.TimeoutExpired as e:
        return False, f"executor_runner timed out after 60 seconds: {e}"


def read_binary_output_file(filepath: Path, dtype: torch.dtype) -> Optional[np.ndarray]:
    """
    Read binary output values from an executor_runner output file.

    Args:
        filepath: Path to the binary output file
        dtype: The torch dtype to interpret the binary data as

    Returns:
        numpy array of values, or None if file doesn't exist or is empty
    """
    if not filepath.exists():
        return None

    # Map torch dtype to numpy dtype
    dtype_map = {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.bfloat16: np.float32,  # bfloat16 is read as float32 after conversion
        torch.int32: np.int32,
        torch.int64: np.int64,
    }

    np_dtype = dtype_map.get(dtype, np.float32)

    try:
        with open(filepath, "rb") as f:
            binary_data = f.read()
            if not binary_data:
                return None
            # For bfloat16, the runtime output is in bfloat16 format (2 bytes per element)
            # We need to read it as uint16 and convert
            if dtype == torch.bfloat16:
                # Read as uint16 (2 bytes per element like bfloat16)
                values_uint16 = np.frombuffer(binary_data, dtype=np.uint16)
                # Convert bfloat16 to float32 by shifting left 16 bits
                values_uint32 = values_uint16.astype(np.uint32) << 16
                values = values_uint32.view(np.float32)
            else:
                values = np.frombuffer(binary_data, dtype=np_dtype)
            return values
    except (FileNotFoundError, ValueError) as e:
        print(f"Error reading binary file {filepath}: {e}")
        return None


def compare_outputs(
    expected: torch.Tensor,
    runtime_output_file: Path,
    dtype: torch.dtype,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
) -> Tuple[bool, Optional[float], Optional[float]]:
    """
    Compare expected PyTorch output with runtime output from binary file.

    Args:
        expected: Expected output tensor from PyTorch
        runtime_output_file: Path to the binary output file from executor_runner
        dtype: The dtype used for the model (needed to parse binary output)
        atol: Absolute tolerance for comparison (if None, uses dtype-specific default)
        rtol: Relative tolerance for comparison (if None, uses dtype-specific default)

    Returns:
        Tuple of (is_close, max_atol, max_rtol)
    """
    # Use dtype-specific tolerances if not specified
    tolerances = DEFAULT_TOLERANCES.get(dtype, DEFAULT_TOLERANCES[torch.float32])
    if atol is None:
        atol = tolerances["atol"]
    if rtol is None:
        rtol = tolerances["rtol"]

    runtime_values = read_binary_output_file(runtime_output_file, dtype)
    if runtime_values is None:
        return False, None, None

    # Flatten expected output and move to CPU for numpy conversion
    # (required when tensor is on MPS device)
    if isinstance(expected, tuple):
        expected_values = np.concatenate(
            [t.detach().cpu().float().flatten().numpy() for t in expected]
        )
    else:
        expected_values = expected.detach().cpu().float().flatten().numpy()

    if len(runtime_values) != len(expected_values):
        return False, None, None

    # Calculate tolerances
    abs_diff = np.abs(runtime_values - expected_values)
    max_atol_val = np.max(abs_diff)

    eps = 1e-8
    denominator = np.maximum(
        np.maximum(np.abs(runtime_values), np.abs(expected_values)), eps
    )
    rel_diff = abs_diff / denominator
    max_rtol_val = np.max(rel_diff)

    is_close = np.allclose(runtime_values, expected_values, atol=atol, rtol=rtol)

    return is_close, max_atol_val, max_rtol_val


# =============================================================================
# Test Class
# =============================================================================


class TestMetalBackendModules(unittest.TestCase):
    """
    Test Metal backend modules export and execution.

    Each test exports a model through the Metal backend and verifies:
    1. The export process completes without errors
    2. The exported program has non-zero buffer size
    3. The runtime output matches the expected PyTorch output
    """

    def _test_module_export(
        self, model_name: str, dtype: torch.dtype = torch.float32
    ) -> None:
        """Generic test for module export."""
        # Check if this model/dtype combination should be skipped
        skip, skip_reason = should_skip_model(model_name, dtype)
        if skip:
            self.skipTest(skip_reason)

        if SKIP_EXPORT_TESTS:
            self.skipTest(SKIP_REASON)

        model, example_inputs = get_model_and_inputs(model_name, dtype=dtype)

        # Export to Metal backend
        executorch_program = export_model_to_metal(model, example_inputs)

        self.assertIsNotNone(
            executorch_program,
            f"{model_name} ({DTYPE_NAMES[dtype]}): Export returned None",
        )
        self.assertGreater(
            len(executorch_program.buffer),
            0,
            f"{model_name} ({DTYPE_NAMES[dtype]}): Exported buffer is empty",
        )

    def _test_module_output_consistency(
        self, model_name: str, dtype: torch.dtype = torch.float32
    ) -> None:
        """
        Test that Metal backend runtime output matches PyTorch output.

        This test:
        1. Exports the model to a .pte file
        2. Runs the model using executor_runner
        3. Compares the runtime output with expected PyTorch output
        """
        # Check if this model/dtype combination should be skipped
        skip, skip_reason = should_skip_model(model_name, dtype)
        if skip:
            self.skipTest(skip_reason)

        if SKIP_RUNTIME_TESTS:
            self.skipTest(SKIP_RUNTIME_REASON)

        model, example_inputs = get_model_and_inputs(model_name, dtype=dtype)
        dtype_name = DTYPE_NAMES[dtype]
        test_subdir_name = f"{model_name}_{dtype_name}"
        model_config = MODULE_REGISTRY.get(model_name, {})

        def run_test_in_directory(test_dir: Path) -> None:
            """Run the actual test logic in the given directory."""
            # Create model output directory: metal_backend_module_outputs/<model_name>_<dtype>/
            model_output_dir = test_dir / test_subdir_name
            model_output_dir.mkdir(parents=True, exist_ok=True)

            # Determine if we should compare to unquantized reference
            # Default to True for quantized models, False otherwise
            compare_to_unquantized = model_config.get(
                "compare_to_unquantized", bool(model_config.get("qlinear"))
            )

            # Export model and get expected output
            pte_path, expected_output = export_model_to_pte(
                model,
                example_inputs,
                model_output_dir,
                model_name,
                compare_to_unquantized=compare_to_unquantized,
                model_config=model_config,
            )

            self.assertTrue(
                pte_path.exists(),
                f"{model_name} ({dtype_name}): PTE file not created at {pte_path}",
            )

            # Run executor_runner with output file
            output_base_path = model_output_dir / "output"
            success, error_msg = run_executor_runner(pte_path, output_base_path)
            self.assertTrue(
                success,
                f"{model_name} ({dtype_name}): executor_runner failed\n{error_msg}",
            )

            # executor_runner writes output files as <output_path>-<index>.bin
            # For single output models, this is output-0.bin
            runtime_output_file = model_output_dir / "output-0.bin"

            self.assertTrue(
                runtime_output_file.exists(),
                f"{model_name} ({dtype_name}): Runtime output file not created at {runtime_output_file}",
            )

            # Get model-specific tolerances (with dtype-specific overrides)
            atol, rtol = get_tolerances_for_model(model_name, dtype)

            is_close, max_atol, max_rtol = compare_outputs(
                expected_output, runtime_output_file, dtype, atol=atol, rtol=rtol
            )

            self.assertTrue(
                is_close,
                f"{model_name} ({dtype_name}): Output mismatch - max_atol={max_atol}, max_rtol={max_rtol}",
            )

        if IS_CI:
            # In CI, use a persistent directory in the current working directory
            TEST_OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
            run_test_in_directory(TEST_OUTPUT_BASE_DIR)
        else:
            # Locally, use a temporary directory that gets cleaned up
            with tempfile.TemporaryDirectory() as tmpdir:
                run_test_in_directory(Path(tmpdir))


# =============================================================================
# Dynamically generate test methods for each module and dtype in MODULE_REGISTRY
# =============================================================================


def _make_export_test(model_name: str, dtype: torch.dtype):
    """Factory function to create an export test method for a given model and dtype."""

    def test_method(self):
        self._test_module_export(model_name, dtype)

    dtype_name = DTYPE_NAMES[dtype]
    test_method.__doc__ = f"Test {model_name} module export with {dtype_name}."
    return test_method


def _make_output_consistency_test(model_name: str, dtype: torch.dtype):
    """Factory function to create an output consistency test method for a given model and dtype."""

    def test_method(self):
        self._test_module_output_consistency(model_name, dtype)

    dtype_name = DTYPE_NAMES[dtype]
    test_method.__doc__ = (
        f"Test {model_name} module output consistency with {dtype_name}."
    )
    return test_method


# Add export and output consistency tests for each module and dtype in the registry
for _model_name in MODULE_REGISTRY:
    for _dtype in DTYPES:
        _dtype_name = DTYPE_NAMES[_dtype]

        # Create export test: test_<model_name>_<dtype>_export
        _export_test_name = f"test_{_model_name}_{_dtype_name}_export"
        setattr(
            TestMetalBackendModules,
            _export_test_name,
            _make_export_test(_model_name, _dtype),
        )

        # Create output consistency test: test_<model_name>_<dtype>_output_consistency
        _consistency_test_name = f"test_{_model_name}_{_dtype_name}_output_consistency"
        setattr(
            TestMetalBackendModules,
            _consistency_test_name,
            _make_output_consistency_test(_model_name, _dtype),
        )


if __name__ == "__main__":
    unittest.main()
