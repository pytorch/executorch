#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for MLX delegate op testing.

This module provides functions to:
1. Save/load tensors to/from binary files (compatible with C++ op_test_runner)
2. Export simple models to .pte files
3. Compare expected vs actual outputs
4. Run the C++ op_test_runner binary
"""

import struct
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


# =============================================================================
# Timeout Support
# =============================================================================

DEFAULT_TEST_TIMEOUT = 300  # 5 minutes default timeout


class TestTimeoutError(Exception):
    """Raised when a test exceeds its timeout."""

    pass


# DType enum values matching C++ op_test_runner
DTYPE_FLOAT32 = 0
DTYPE_FLOAT16 = 1
DTYPE_INT32 = 2
DTYPE_INT64 = 3
DTYPE_BFLOAT16 = 4


# =============================================================================
# Tolerance Presets by DType
# =============================================================================

# Default tolerance presets for different data types.
# These are based on the precision characteristics of each dtype:
# - FP32: ~7 decimal digits of precision
# - FP16: ~3-4 decimal digits of precision
# - BF16: ~2-3 decimal digits of precision (same exponent range as FP32)
TOLERANCE_PRESETS = {
    torch.float32: {"rtol": 1e-5, "atol": 1e-5},
    torch.float16: {"rtol": 1e-3, "atol": 1e-3},
    torch.bfloat16: {"rtol": 1e-2, "atol": 1e-2},
    # Integer types should match exactly
    torch.int32: {"rtol": 0, "atol": 0},
    torch.int64: {"rtol": 0, "atol": 0},
}


def get_tolerance_for_dtype(dtype: torch.dtype) -> Tuple[float, float]:
    """
    Get appropriate (rtol, atol) tolerances for a given dtype.

    Args:
        dtype: The torch dtype to get tolerances for.

    Returns:
        (rtol, atol) tuple with appropriate tolerances for the dtype.
    """
    if dtype in TOLERANCE_PRESETS:
        preset = TOLERANCE_PRESETS[dtype]
        return preset["rtol"], preset["atol"]
    # Default to FP32 tolerances for unknown types
    return 1e-5, 1e-5


def get_tolerance_for_dtypes(dtypes: List[torch.dtype]) -> Tuple[float, float]:
    """
    Get tolerances that work for a list of dtypes (uses the loosest tolerances).

    Args:
        dtypes: List of torch dtypes.

    Returns:
        (rtol, atol) tuple with tolerances that accommodate all dtypes.
    """
    if not dtypes:
        return 1e-5, 1e-5

    max_rtol = 0.0
    max_atol = 0.0
    for dtype in dtypes:
        rtol, atol = get_tolerance_for_dtype(dtype)
        max_rtol = max(max_rtol, rtol)
        max_atol = max(max_atol, atol)

    return max_rtol, max_atol


def torch_dtype_to_bin_dtype(dtype: torch.dtype) -> int:
    """Convert torch dtype to binary file dtype enum value."""
    mapping = {
        torch.float32: DTYPE_FLOAT32,
        torch.float16: DTYPE_FLOAT16,
        torch.int32: DTYPE_INT32,
        torch.int64: DTYPE_INT64,
        torch.bfloat16: DTYPE_BFLOAT16,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def bin_dtype_to_torch_dtype(dtype_val: int) -> torch.dtype:
    """Convert binary file dtype enum value to torch dtype."""
    mapping = {
        DTYPE_FLOAT32: torch.float32,
        DTYPE_FLOAT16: torch.float16,
        DTYPE_INT32: torch.int32,
        DTYPE_INT64: torch.int64,
        DTYPE_BFLOAT16: torch.bfloat16,
    }
    if dtype_val not in mapping:
        raise ValueError(f"Unknown dtype value: {dtype_val}")
    return mapping[dtype_val]


def save_tensors_to_bin(tensors: List[torch.Tensor], path: Union[str, Path]) -> None:
    """
    Save a list of tensors to a binary file.

    Binary format:
    - 4 bytes: number of tensors (uint32)
    For each tensor:
      - 4 bytes: dtype enum (uint32)
      - 4 bytes: number of dimensions (uint32)
      - 4 bytes * ndim: shape (int32 each)
      - N bytes: tensor data
    """
    path = Path(path)

    with open(path, "wb") as f:
        # Write number of tensors
        f.write(struct.pack("I", len(tensors)))

        for tensor in tensors:
            # Ensure contiguous
            tensor = tensor.contiguous()

            # Write dtype
            dtype_val = torch_dtype_to_bin_dtype(tensor.dtype)
            f.write(struct.pack("I", dtype_val))

            # Write ndim
            f.write(struct.pack("I", tensor.dim()))

            # Write shape
            for s in tensor.shape:
                f.write(struct.pack("i", s))

            # Write data - bf16 needs special handling since numpy doesn't support it
            if tensor.dtype == torch.bfloat16:
                # View bf16 as uint16 to preserve raw bytes
                f.write(tensor.view(torch.uint16).numpy().tobytes())
            else:
                f.write(tensor.numpy().tobytes())


def load_tensors_from_bin(path: Union[str, Path]) -> List[torch.Tensor]:
    """
    Load a list of tensors from a binary file.
    """
    path = Path(path)

    # Mapping from torch dtype to numpy dtype
    np_dtype_map = {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.int32: np.int32,
        torch.int64: np.int64,
        # bfloat16 needs special handling - read as uint16
    }

    # Element size for each dtype
    elem_size_map = {
        torch.float32: 4,
        torch.float16: 2,
        torch.int32: 4,
        torch.int64: 8,
        torch.bfloat16: 2,
    }

    tensors = []
    with open(path, "rb") as f:
        # Read number of tensors
        num_tensors = struct.unpack("I", f.read(4))[0]

        for _ in range(num_tensors):
            # Read dtype
            dtype_val = struct.unpack("I", f.read(4))[0]
            dtype = bin_dtype_to_torch_dtype(dtype_val)

            # Read ndim
            ndim = struct.unpack("I", f.read(4))[0]

            # Read shape
            shape = []
            for _ in range(ndim):
                shape.append(struct.unpack("i", f.read(4))[0])

            # Read data
            numel = 1
            for s in shape:
                numel *= s

            elem_size = elem_size_map[dtype]
            data_bytes = f.read(numel * elem_size)

            # Convert to tensor
            if dtype == torch.bfloat16:
                # Read as uint16 and view as bfloat16
                arr = np.frombuffer(data_bytes, dtype=np.uint16).reshape(shape)
                tensor = torch.tensor(arr).view(torch.bfloat16)
            else:
                arr = np.frombuffer(data_bytes, dtype=np_dtype_map[dtype]).reshape(
                    shape
                )
                tensor = torch.from_numpy(arr.copy())

            tensors.append(tensor)

    return tensors


def export_model_to_pte(
    model: torch.nn.Module,
    example_inputs: Tuple[torch.Tensor, ...],
    output_path: Union[str, Path],
    use_fp16: bool = False,
    dynamic_shapes: Optional[Dict] = None,
    verbose: bool = False,
) -> None:
    """
    Export a PyTorch model to a .pte file using the MLX delegate.

    Args:
        model: The PyTorch model to export.
        example_inputs: Example inputs for tracing.
        output_path: Path to save the .pte file.
        use_fp16: Whether to use FP16 precision.
        dynamic_shapes: Optional dynamic shapes specification for torch.export.
            Example: {0: {0: Dim("batch", min=1, max=32)}} for dynamic batch on first input.
        verbose: Whether to print the exported program for debugging.
    """
    import executorch.exir as exir
    from executorch.backends.apple.mlx import MLXPartitioner
    from executorch.exir.backend.backend_details import CompileSpec
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from torch.export import export

    model = model.eval()

    # Export with torch.export
    exported_program = export(
        model, example_inputs, dynamic_shapes=dynamic_shapes, strict=True
    )

    # Print exported program if verbose
    if verbose:
        print("\n" + "=" * 60)
        print("EXPORTED PROGRAM (torch.export)")
        print("=" * 60)
        print(exported_program)

    # Lower to edge and delegate to MLX
    compile_specs = [CompileSpec("use_fp16", bytes([use_fp16]))]
    edge_program = exir.to_edge_transform_and_lower(
        exported_program,
        partitioner=[MLXPartitioner(compile_specs=compile_specs)],
    )

    # Print edge program if verbose
    if verbose:
        print("\n" + "=" * 60)
        print("EDGE PROGRAM (after decomposition)")
        print("=" * 60)
        print(edge_program.exported_program())

    # Export to ExecuTorch
    executorch_program = edge_program.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=True)
    )

    # Save to file
    output_path = Path(output_path)
    with open(output_path, "wb") as f:
        f.write(executorch_program.buffer)


def inspect_pte_file(pte_path: Union[str, Path]) -> Dict:
    """
    Inspect a PTE file and return the MLX graph information.

    Returns:
        Dictionary with MLX graph details
    """
    from executorch.backends.apple.mlx.pte_inspector import (
        extract_delegate_payload,
        parse_mlx_payload,
    )

    pte_path = Path(pte_path)
    pte_data = pte_path.read_bytes()

    # Extract MLX delegate payload
    payload = extract_delegate_payload(pte_data, "MLXBackend")
    if payload is None:
        return {"error": "Could not extract MLX delegate payload"}

    # Parse the MLX payload
    mlx_data = parse_mlx_payload(payload)
    return mlx_data


def print_mlx_graph_summary(pte_path: Union[str, Path]) -> None:
    """
    Print a human-readable summary of the MLX graph in a PTE file.

    This function uses the pte_inspector module to display the MLX graph.
    """
    from executorch.backends.apple.mlx.pte_inspector import show_mlx_instructions

    pte_path = Path(pte_path)
    pte_data = pte_path.read_bytes()

    show_mlx_instructions(pte_data)


def compare_outputs(
    expected: List[torch.Tensor],
    actual: List[torch.Tensor],
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> Tuple[bool, str]:
    """
    Compare expected and actual outputs using torch.allclose.

    Returns:
        (passed, message) tuple
    """
    if len(expected) != len(actual):
        return (
            False,
            f"Output count mismatch: expected {len(expected)}, got {len(actual)}",
        )

    for i, (exp, act) in enumerate(zip(expected, actual)):
        if exp.shape != act.shape:
            return (
                False,
                f"Output {i} shape mismatch: expected {exp.shape}, got {act.shape}",
            )

        if exp.dtype != act.dtype:
            # Convert both to float32 for comparison
            exp = exp.float()
            act = act.float()

        if not torch.allclose(exp, act, rtol=rtol, atol=atol):
            max_diff = (exp - act).abs().max().item()
            mean_diff = (exp - act).abs().mean().item()
            return False, (
                f"Output {i} values do not match:\n"
                f"  max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}\n"
                f"  rtol={rtol}, atol={atol}\n"
                f"  expected[:5]={exp.flatten()[:5].tolist()}\n"
                f"  actual[:5]={act.flatten()[:5].tolist()}"
            )

    return True, "All outputs match"


def find_executorch_root() -> Path:
    """Find the executorch root directory."""
    test_dir = Path(__file__).parent

    # Walk up to find the executorch root (has CMakeLists.txt and backends dir at root)
    executorch_root = test_dir
    for _ in range(10):  # Max 10 levels up
        if (executorch_root / "CMakeLists.txt").exists() and (
            executorch_root / "backends"
        ).exists():
            # Check if we're in src/executorch (editable install)
            if (
                executorch_root.name == "executorch"
                and executorch_root.parent.name == "src"
            ):
                executorch_root = executorch_root.parent.parent
            break
        executorch_root = executorch_root.parent

    return executorch_root


def find_build_dir() -> Optional[Path]:
    """Find the cmake build directory containing op_test_runner."""
    executorch_root = find_executorch_root()

    # Check common build locations
    candidates = [
        executorch_root / "cmake-out-mlx",
        executorch_root / "cmake-out",
        executorch_root / "build",
    ]

    for candidate in candidates:
        runner_path = (
            candidate / "backends" / "apple" / "mlx" / "test" / "op_test_runner"
        )
        if runner_path.exists():
            return candidate

    # Return first candidate that exists as a directory (for rebuild)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    return None


def find_op_test_runner() -> Path:
    """Find the op_test_runner binary."""
    executorch_root = find_executorch_root()

    # Check common build locations
    candidates = [
        executorch_root
        / "cmake-out-mlx"
        / "backends"
        / "apple"
        / "mlx"
        / "test"
        / "op_test_runner",
        executorch_root
        / "cmake-out"
        / "backends"
        / "apple"
        / "mlx"
        / "test"
        / "op_test_runner",
        executorch_root
        / "build"
        / "backends"
        / "apple"
        / "mlx"
        / "test"
        / "op_test_runner",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find op_test_runner binary. Tried:\n"
        + "\n".join(f"  - {c}" for c in candidates)
        + "\n\nBuild with: cd cmake-out && cmake --build . --target op_test_runner"
    )


def rebuild_op_test_runner(verbose: bool = False) -> bool:
    """
    Rebuild the op_test_runner binary using cmake.

    Args:
        verbose: Whether to print build output.

    Returns:
        True if build succeeded, False otherwise.
    """
    build_dir = find_build_dir()
    if build_dir is None:
        print("Error: Could not find cmake build directory.")
        print("Make sure you have run cmake configuration first.")
        return False

    print(f"Rebuilding op_test_runner in {build_dir}...")

    cmd = ["cmake", "--build", str(build_dir), "--target", "op_test_runner", "-j8"]

    if verbose:
        print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=not verbose,
        text=True,
    )

    if result.returncode != 0:
        print(f"Build failed with exit code {result.returncode}")
        if not verbose and result.stderr:
            print(f"stderr: {result.stderr}")
        if not verbose and result.stdout:
            print(f"stdout: {result.stdout}")
        return False

    print("Build succeeded.")
    return True


def run_cpp_test_runner(
    pte_path: Path,
    input_path: Path,
    output_path: Path,
    verbose: bool = False,
    timeout: Optional[int] = None,
) -> bool:
    """
    Run the C++ op_test_runner binary.

    Args:
        pte_path: Path to the .pte model file.
        input_path: Path to input .bin file.
        output_path: Path to write output .bin file.
        verbose: Whether to print verbose output.
        timeout: Timeout in seconds. None means use DEFAULT_TEST_TIMEOUT.

    Returns:
        True if execution succeeded, False otherwise.
    """
    if timeout is None:
        timeout = DEFAULT_TEST_TIMEOUT

    runner = find_op_test_runner()

    cmd = [
        str(runner),
        "--pte",
        str(pte_path),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]
    if verbose:
        cmd.append("--verbose")

    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: C++ runner exceeded {timeout}s timeout")
        return False

    if result.returncode != 0:
        print(f"FAILED: {result.stderr}")
        print(f"stdout: {result.stdout}")
        return False

    print(f"C++ binary output: {result.stdout.strip()}")
    return True


# =============================================================================
# Cleanup Utilities
# =============================================================================

# Files that are generated during tests and can be safely cleaned up
GENERATED_TEST_FILES = [
    "model.pte",
    "input.bin",
    "expected_output.bin",
    "actual_output.bin",
]


def clean_test_outputs(
    test_names: Optional[List[str]] = None, verbose: bool = False
) -> int:
    """
    Clean up generated test output files.

    Args:
        test_names: Optional list of test names to clean. If None, cleans all tests.
        verbose: Whether to print verbose output.

    Returns:
        Number of files removed.
    """
    test_dir = Path(__file__).parent / "op_tests"
    if not test_dir.exists():
        if verbose:
            print(f"Test directory does not exist: {test_dir}")
        return 0

    files_removed = 0

    # Get directories to clean
    if test_names:
        dirs_to_clean = [
            test_dir / name for name in test_names if (test_dir / name).exists()
        ]
    else:
        dirs_to_clean = [d for d in test_dir.iterdir() if d.is_dir()]

    for subdir in dirs_to_clean:
        for filename in GENERATED_TEST_FILES:
            filepath = subdir / filename
            if filepath.exists():
                if verbose:
                    print(f"Removing: {filepath}")
                filepath.unlink()
                files_removed += 1

        # Remove empty directories
        if subdir.exists() and not any(subdir.iterdir()):
            if verbose:
                print(f"Removing empty directory: {subdir}")
            subdir.rmdir()

    return files_removed


def get_test_output_size(test_names: Optional[List[str]] = None) -> int:
    """
    Get total size of generated test output files in bytes.

    Args:
        test_names: Optional list of test names to check. If None, checks all tests.

    Returns:
        Total size in bytes.
    """
    test_dir = Path(__file__).parent / "op_tests"
    if not test_dir.exists():
        return 0

    total_size = 0

    # Get directories to check
    if test_names:
        dirs_to_check = [
            test_dir / name for name in test_names if (test_dir / name).exists()
        ]
    else:
        dirs_to_check = [d for d in test_dir.iterdir() if d.is_dir()]

    for subdir in dirs_to_check:
        for filename in GENERATED_TEST_FILES:
            filepath = subdir / filename
            if filepath.exists():
                total_size += filepath.stat().st_size

    return total_size


# =============================================================================
# Test Registry
# =============================================================================

# Global registry: maps base_name -> (test_class, get_test_configs method)
# Tests are instantiated lazily when actually run, not at import time
_TEST_REGISTRY: Dict[str, type] = {}


def register_test(test_class: type) -> type:
    """
    Class decorator to register a test class.

    The test class must have:
    - A class attribute `name` (str) - the base test name
    - A class method `get_test_configs()` that returns a list of OpTestCase instances

    Test instances are created LAZILY when tests are actually run, not at import time.
    This avoids creating random tensors at import time and keeps memory usage low.

    Example:
        @register_test
        class AddTest(OpTestCase):
            name = "add"

            @classmethod
            def get_test_configs(cls) -> List["OpTestCase"]:
                return [
                    cls(),  # default config
                    cls(scalar=2.5),  # scalar variant
                ]
    """
    if not hasattr(test_class, "name"):
        raise ValueError(
            f"Test class {test_class.__name__} must have a 'name' attribute"
        )

    base_name = test_class.name
    _TEST_REGISTRY[base_name] = test_class

    return test_class


def get_registered_tests() -> Dict[str, List[Tuple[str, "OpTestCase"]]]:
    """
    Get all registered tests with their configurations.

    Returns dict mapping base_name -> list of (config_name, test_instance).
    Test instances are created fresh each time this is called.
    """
    result = {}
    for base_name, test_class in _TEST_REGISTRY.items():
        if hasattr(test_class, "get_test_configs"):
            configs = test_class.get_test_configs()
        else:
            configs = [test_class()]
        result[base_name] = [(cfg.name, cfg) for cfg in configs]
    return result


def get_test_names() -> List[str]:
    """Get list of registered base test names."""
    return list(_TEST_REGISTRY.keys())


def get_all_test_configs() -> List[Tuple[str, "OpTestCase"]]:
    """
    Get flat list of all (config_name, test_instance) tuples.

    Test instances are created fresh each time this is called.
    """
    result = []
    for _base_name, test_class in _TEST_REGISTRY.items():
        if hasattr(test_class, "get_test_configs"):
            configs = test_class.get_test_configs()
        else:
            configs = [test_class()]
        result.extend((cfg.name, cfg) for cfg in configs)
    return result


class OpTestCase:
    """
    Base class for op test cases.

    Subclasses should implement:
    - name: str - test name
    - create_model() -> nn.Module
    - create_inputs() -> Tuple[torch.Tensor, ...]

    Optionally override:
    - get_dynamic_shapes() -> Optional[Dict] - for dynamic shape testing
    - create_test_inputs() -> Tuple[torch.Tensor, ...] - inputs for testing (may differ from export inputs)
    """

    name: str = "base_test"
    rtol: float = 1e-5
    atol: float = 1e-5
    use_fp16: bool = False
    seed: int = 42  # Default seed for reproducibility
    timeout: int = DEFAULT_TEST_TIMEOUT  # Timeout in seconds
    skip_comparison: bool = False  # Skip output comparison (for pattern-only tests)
    skip_comparison_reason: str = ""  # Reason for skipping comparison

    def _set_seed(self) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(self.seed)

    def create_model(self) -> torch.nn.Module:
        raise NotImplementedError

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        """Create inputs for export (tracing)."""
        raise NotImplementedError

    def create_test_inputs(self) -> Tuple[torch.Tensor, ...]:
        """Create inputs for testing. Override for dynamic shape tests."""
        return self.create_inputs()

    def get_dynamic_shapes(self) -> Optional[Dict]:
        """Return dynamic shapes specification for torch.export, or None for static shapes."""
        return None

    def get_test_dir(self) -> Path:
        """Get the directory for this test's files."""
        test_dir = Path(__file__).parent / "op_tests" / self.name
        test_dir.mkdir(parents=True, exist_ok=True)
        return test_dir

    def generate_test_files(self, verbose: bool = False) -> Tuple[Path, Path, Path]:
        """
        Generate .pte, input.bin, and expected_output.bin files.

        Args:
            verbose: Whether to print the exported program for debugging.

        Returns:
            (pte_path, input_path, expected_output_path)
        """
        test_dir = self.get_test_dir()

        pte_path = test_dir / "model.pte"
        input_path = test_dir / "input.bin"
        expected_path = test_dir / "expected_output.bin"

        # Set seed for reproducibility
        self._set_seed()

        # Create model and inputs
        model = self.create_model()
        export_inputs = self.create_inputs()

        # Set seed again before creating test inputs (in case they differ)
        self._set_seed()
        test_inputs = self.create_test_inputs()

        # Get expected outputs using test inputs
        model.eval()
        with torch.no_grad():
            if isinstance(test_inputs, torch.Tensor):
                test_inputs = (test_inputs,)
            expected_outputs = model(*test_inputs)
            if isinstance(expected_outputs, torch.Tensor):
                expected_outputs = [expected_outputs]
            else:
                expected_outputs = list(expected_outputs)

        # Export model with export inputs (and potentially dynamic shapes)
        print(f"Exporting model to {pte_path}")
        if isinstance(export_inputs, torch.Tensor):
            export_inputs = (export_inputs,)

        dynamic_shapes = self.get_dynamic_shapes()
        if dynamic_shapes:
            print(f"  Using dynamic shapes: {dynamic_shapes}")

        export_model_to_pte(
            model,
            export_inputs,
            pte_path,
            use_fp16=self.use_fp16,
            dynamic_shapes=dynamic_shapes,
            verbose=verbose,
        )

        # Save test inputs
        print(f"Saving inputs to {input_path}")
        if isinstance(test_inputs, torch.Tensor):
            test_inputs = [test_inputs]
        else:
            test_inputs = list(test_inputs)
        save_tensors_to_bin(test_inputs, input_path)

        # Save expected outputs
        print(f"Saving expected outputs to {expected_path}")
        save_tensors_to_bin(expected_outputs, expected_path)

        return pte_path, input_path, expected_path

    def compare_with_actual(
        self, actual_output_path: Union[str, Path], use_dtype_tolerances: bool = False
    ) -> Tuple[bool, str]:
        """
        Compare actual outputs with expected outputs.

        Args:
            actual_output_path: Path to the actual output file.
            use_dtype_tolerances: If True, uses tolerance presets based on output dtypes
                instead of the test's rtol/atol values.
        """
        test_dir = self.get_test_dir()
        expected_path = test_dir / "expected_output.bin"

        expected = load_tensors_from_bin(expected_path)
        actual = load_tensors_from_bin(actual_output_path)

        # Determine tolerances
        if use_dtype_tolerances:
            # Use dtype-based tolerances (loosest tolerance across all output dtypes)
            output_dtypes = [t.dtype for t in expected]
            rtol, atol = get_tolerance_for_dtypes(output_dtypes)
        else:
            rtol, atol = self.rtol, self.atol

        return compare_outputs(expected, actual, rtol=rtol, atol=atol)

    def run_test(self, verbose: bool = False, timeout: Optional[int] = None) -> bool:
        """
        Run the full test: generate files, run C++, compare outputs.

        Args:
            verbose: Whether to print verbose output.
            timeout: Timeout in seconds. None means use self.timeout.

        Returns:
            True if test passed, False otherwise.
        """
        if timeout is None:
            timeout = self.timeout

        print(f"\n{'='*60}")
        print(f"Running test: {self.name}")
        print(f"{'='*60}\n")

        # Generate test files
        print("Step 1: Generating test files...")
        pte_path, input_path, expected_path = self.generate_test_files(verbose=verbose)

        # Print MLX graph summary
        print_mlx_graph_summary(pte_path)

        # Run C++ binary
        print("Step 2: Running C++ binary...")
        actual_path = self.get_test_dir() / "actual_output.bin"
        if not run_cpp_test_runner(
            pte_path, input_path, actual_path, verbose=verbose, timeout=timeout
        ):
            return False

        # Compare outputs (or skip if configured)
        print("\nStep 3: Comparing outputs...")
        if self.skip_comparison:
            reason = self.skip_comparison_reason or "skip_comparison=True"
            print(f"NOTE: Output comparison skipped ({reason})")
            print("✓ PASSED (runtime execution succeeded)")
            return True

        passed, message = self.compare_with_actual(actual_path)

        if passed:
            print(f"✓ PASSED: {message}")
        else:
            print(f"✗ FAILED: {message}")

        return passed


# =============================================================================
# Common CLI helper for op tests
# =============================================================================


def run_op_test_main(
    test_factory,
    description: str,
    add_args_fn=None,
):
    """
    Common main() function for op tests.

    This handles the common argparse setup, rebuild logic, and generate/compare/run
    action handling that is shared across all op tests.

    Args:
        test_factory: A callable that takes parsed args (argparse.Namespace) and
            returns an OpTestCase instance.
        description: Description for the argparse help message.
        add_args_fn: Optional callable that takes a parser and adds test-specific
            arguments. Signature: add_args_fn(parser) -> None
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "action",
        choices=["generate", "compare", "run"],
        help="Action to perform: generate (create test files), compare (compare outputs), run (full test)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the C++ test runner before running",
    )

    # Add test-specific arguments
    if add_args_fn is not None:
        add_args_fn(parser)

    args = parser.parse_args()

    # Rebuild if requested
    if args.rebuild:
        if not rebuild_op_test_runner(verbose=args.verbose):
            sys.exit(1)

    # Create test case from factory
    test = test_factory(args)

    if args.action == "generate":
        pte_path, input_path, expected_path = test.generate_test_files(
            verbose=args.verbose
        )
        print("\nGenerated files:")
        print(f"  PTE:      {pte_path}")
        print(f"  Input:    {input_path}")
        print(f"  Expected: {expected_path}")
        print_mlx_graph_summary(pte_path)

    elif args.action == "compare":
        actual_path = test.get_test_dir() / "actual_output.bin"
        if not actual_path.exists():
            print(f"Error: {actual_path} not found. Run the C++ binary first.")
            sys.exit(1)

        passed, message = test.compare_with_actual(actual_path)
        if passed:
            print(f"✓ PASSED: {message}")
        else:
            print(f"✗ FAILED: {message}")
        sys.exit(0 if passed else 1)

    elif args.action == "run":
        passed = test.run_test(verbose=args.verbose)
        sys.exit(0 if passed else 1)
