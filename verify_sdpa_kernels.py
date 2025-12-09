#!/usr/bin/env python3
"""
SDPA Kernel Verification Script

This script reads the logged inputs and outputs from custom SDPA kernels
(flash_attention and efficient_attention) and compares them against PyTorch's
ground truth SDPA implementation.

Usage:
    python verify_sdpa_kernels.py [--log-dir /tmp] [--call-id 0] [--tolerance 1e-3]
"""

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TensorInfo:
    """Information about a tensor loaded from disk"""

    data: torch.Tensor
    shape: tuple
    strides: tuple
    dtype: torch.dtype


def parse_meta_file(meta_path: str) -> Dict[str, any]:
    """Parse metadata file to get tensor information"""
    meta = {}
    with open(meta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(": ", 1)
            meta[key] = value

    # Parse shape
    shape = tuple(map(int, meta["shape"].split(",")))

    # Parse strides
    strides = tuple(map(int, meta["strides"].split(",")))

    # Parse dtype size
    dtype_size = int(meta["dtype_size"])

    # Get the C++ dtype string if available
    cpp_dtype_str = meta.get("dtype", "")

    # Map dtype size to numpy/torch dtype
    # Check C++ type name to distinguish between float16 and bfloat16
    if dtype_size == 4:
        numpy_dtype = np.float32
        torch_dtype = torch.float32
        is_bfloat16 = False
    elif dtype_size == 2:
        # Check if it's bfloat16 by looking at the C++ type name
        # __nv_bfloat16 will have "bfloat" in its mangled name
        if "bfloat" in cpp_dtype_str.lower() or "__nv_bfloat16" in cpp_dtype_str:
            # BFloat16 - numpy doesn't support it natively,
            # we'll read as uint16 and convert via torch
            numpy_dtype = np.uint16  # Read raw bytes as uint16
            torch_dtype = torch.bfloat16
            is_bfloat16 = True
        else:
            # Assume float16 (half)
            numpy_dtype = np.float16
            torch_dtype = torch.float16
            is_bfloat16 = False
    else:
        raise ValueError(f"Unsupported dtype size: {dtype_size}")

    return {
        "shape": shape,
        "strides": strides,
        "dtype": numpy_dtype,
        "torch_dtype": torch_dtype,
        "dtype_size": dtype_size,
        "is_bfloat16": dtype_size == 2
        and ("bfloat" in cpp_dtype_str.lower() or "__nv_bfloat16" in cpp_dtype_str),
        "cpp_dtype": cpp_dtype_str,
    }


def load_tensor(base_path: str, tensor_name: str) -> Optional[TensorInfo]:
    """Load a tensor from disk"""
    meta_path = os.path.join(base_path, f"{tensor_name}_meta.txt")
    data_path = os.path.join(base_path, f"{tensor_name}_data.bin")

    if not os.path.exists(meta_path) or not os.path.exists(data_path):
        return None

    # Parse metadata
    meta = parse_meta_file(meta_path)
    shape = meta["shape"]
    numpy_dtype = meta["dtype"]
    torch_dtype = meta["torch_dtype"]
    is_bfloat16 = meta.get("is_bfloat16", False)
    cpp_dtype = meta.get("cpp_dtype", "")

    print(f"Tensor: {tensor_name}")
    print(f"  numpy dtype: {numpy_dtype}")
    print(f"  torch dtype: {torch_dtype}")
    print(f"  is_bfloat16: {is_bfloat16}")
    print(f"  cpp_dtype: {cpp_dtype}")

    # Load binary data
    data = np.fromfile(data_path, dtype=numpy_dtype)

    # Reshape to original shape
    data = data.reshape(shape)

    # Convert to PyTorch tensor
    if is_bfloat16:
        # For BFloat16: we read the raw bytes as uint16,
        # now we need to view them as bfloat16
        # PyTorch can do this conversion
        tensor = torch.from_numpy(data.astype(np.int16)).view(torch.bfloat16)
        print(f"  Converted from uint16 to bfloat16")
    else:
        tensor = torch.from_numpy(data)
        # Cast to the correct torch dtype if needed
        if tensor.dtype != torch_dtype:
            tensor = tensor.to(torch_dtype)

    print(f"  Final tensor dtype: {tensor.dtype}")

    return TensorInfo(
        data=tensor, shape=shape, strides=meta["strides"], dtype=torch_dtype
    )


def load_call_info(base_path: str) -> Dict[str, any]:
    """Load call information from disk"""
    info_path = os.path.join(base_path, "call_info.txt")

    info = {}
    with open(info_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(": ", 1)

            if key == "scale_factor":
                info[key] = float(value)
            elif key == "is_causal":
                info[key] = value.lower() == "true"
            elif key == "call_id":
                info[key] = int(value)
            else:
                info[key] = value

    return info


def compute_pytorch_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor],
    scale_factor: float,
    is_causal: bool,
    device: str = "cuda",
) -> torch.Tensor:
    """Compute ground truth using PyTorch's SDPA"""

    print("Query shape:", query.shape)
    print("Query data:", query)

    print("Key shape:", key.shape)
    print("Key data:", key)

    print("Value shape:", value.shape)
    print("Value data:", value)

    # if attn_bias is not None:
    #     print("Attn bias shape:", attn_bias.shape)
    #     print("Attn bias data:", attn_bias)

    # Move tensors to device
    query = query.to(device)
    key = key.to(device)
    value = value.to(device)

    # Compute SDPA using PyTorch
    with torch.no_grad():
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_bias,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale_factor,
        )

    return output


def compare_tensors(
    actual: torch.Tensor,
    expected: torch.Tensor,
    name: str = "tensor",
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> Dict[str, any]:
    """Compare two tensors and return statistics"""

    # Ensure same device
    if actual.device != expected.device:
        actual = actual.to(expected.device)

    # Ensure same dtype
    if actual.dtype != expected.dtype:
        actual = actual.to(expected.dtype)

    # Calculate differences
    diff = torch.abs(actual - expected)
    rel_diff = diff / (torch.abs(expected) + 1e-8)

    # Calculate statistics
    stats = {
        "name": name,
        "shape": tuple(actual.shape),
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "max_rel_diff": rel_diff.max().item(),
        "mean_rel_diff": rel_diff.mean().item(),
        "allclose": torch.allclose(actual, expected, rtol=rtol, atol=atol),
        "rtol_used": rtol,
        "atol_used": atol,
    }

    # Find location of maximum difference
    max_idx = torch.argmax(diff)
    max_idx_unraveled = np.unravel_index(max_idx.cpu().numpy(), actual.shape)
    stats["max_diff_location"] = max_idx_unraveled
    stats["actual_at_max"] = actual.flatten()[max_idx].item()
    stats["expected_at_max"] = expected.flatten()[max_idx].item()

    # Calculate percentage of elements within tolerance
    close_mask = torch.isclose(actual, expected, rtol=rtol, atol=atol)
    stats["pct_within_tolerance"] = (close_mask.sum().item() / close_mask.numel()) * 100

    return stats


def print_comparison_stats(stats: Dict[str, any]):
    """Pretty print comparison statistics"""
    print(f"\n{'='*80}")
    print(f"Comparison Results: {stats['name']}")
    print(f"{'='*80}")
    print(f"Shape: {stats['shape']}")
    print(f"Tolerance used: rtol={stats['rtol_used']}, atol={stats['atol_used']}")
    print(f"\nAbsolute Differences:")
    print(f"  Max:  {stats['max_abs_diff']:.6e}")
    print(f"  Mean: {stats['mean_abs_diff']:.6e}")
    print(f"\nRelative Differences:")
    print(f"  Max:  {stats['max_rel_diff']:.6e}")
    print(f"  Mean: {stats['mean_rel_diff']:.6e}")
    print(f"\nTolerance Check:")
    print(f"  Within tolerance: {stats['pct_within_tolerance']:.2f}%")
    print(f"  All close: {'✓ PASS' if stats['allclose'] else '✗ FAIL'}")
    print(f"\nMaximum Difference Location:")
    print(f"  Index: {stats['max_diff_location']}")
    print(f"  Actual value:   {stats['actual_at_max']:.6f}")
    print(f"  Expected value: {stats['expected_at_max']:.6f}")
    print(f"{'='*80}\n")


def verify_kernel_call(
    base_path: str, rtol: float = 1e-3, atol: float = 1e-5, device: str = "cuda"
) -> bool:
    """Verify a single kernel call"""

    print(f"\n{'*'*80}")
    print(f"Verifying kernel call: {base_path}")
    print(f"{'*'*80}")

    # Load call info
    info = load_call_info(base_path)
    print(f"\nKernel: {info['kernel_name']}")
    print(f"Call ID: {info['call_id']}")
    print(f"Scale factor: {info['scale_factor']}")
    print(f"Is causal: {info['is_causal']}")

    # Load tensors
    print("\nLoading tensors...")
    query_info = load_tensor(base_path, "query")
    key_info = load_tensor(base_path, "key")
    value_info = load_tensor(base_path, "value")
    attn_bias_info = load_tensor(base_path, "attn_bias")  # May be None
    output_info = load_tensor(base_path, "output")

    if query_info is None or key_info is None or value_info is None:
        print("ERROR: Failed to load required tensors")
        return False

    print(f"  Query shape: {query_info.shape}")
    print(f"  Key shape: {key_info.shape}")
    print(f"  Value shape: {value_info.shape}")
    if attn_bias_info is not None:
        print(f"  Attn bias shape: {attn_bias_info.shape}")

    if output_info is not None:
        print(f"  Output shape: {output_info.shape}")

    # Compute PyTorch ground truth
    print("\nComputing PyTorch SDPA ground truth...")
    try:
        pytorch_output = compute_pytorch_sdpa(
            query_info.data,
            key_info.data,
            value_info.data,
            attn_bias_info.data if attn_bias_info is not None else None,
            info["scale_factor"],
            info["is_causal"],
            device=device,
        )
    except Exception as e:
        print(f"ERROR: Failed to compute PyTorch SDPA: {e}")
        import traceback

        traceback.print_exc()
        return False

    if output_info is not None:
        # Compare outputs
        print("\nComparing outputs...")
        stats = compare_tensors(
            output_info.data.to(device),
            pytorch_output,
            name=f"{info['kernel_name']}_output",
            rtol=rtol,
            atol=atol,
        )

        print_comparison_stats(stats)

        # Save detailed comparison to file
        comparison_path = os.path.join(base_path, "comparison_results.txt")
        with open(comparison_path, "w") as f:
            f.write(f"Kernel: {info['kernel_name']}\n")
            f.write(f"Call ID: {info['call_id']}\n")
            f.write(f"{'='*80}\n\n")

            for key, value in stats.items():
                f.write(f"{key}: {value}\n")

            f.write(f"\n{'='*80}\n")
            f.write(f"Result: {'PASS' if stats['allclose'] else 'FAIL'}\n")

        print(f"Detailed comparison saved to: {comparison_path}")

        return stats["allclose"]
    else:
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Verify SDPA kernel outputs against PyTorch ground truth"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="/tmp",
        help="Directory containing logged SDPA calls (default: /tmp)",
    )
    parser.add_argument(
        "--call-id",
        type=int,
        default=None,
        help="Specific call ID to verify (default: verify all)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for comparison (default: 1e-3)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for comparison (default: 1e-5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for PyTorch SDPA (default: cuda)",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        choices=["flash_attention", "efficient_attention", "all"],
        default="all",
        help="Which kernel to verify (default: all)",
    )

    args = parser.parse_args()

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Find all logged calls
    pattern = os.path.join(args.log_dir, "sdpa_debug_*")
    all_dirs = sorted(glob.glob(pattern))

    if not all_dirs:
        print(f"No logged SDPA calls found in {args.log_dir}")
        print(f"Expected pattern: {pattern}")
        return

    print(f"Found {len(all_dirs)} logged SDPA calls")

    # Filter by call_id if specified
    if args.call_id is not None:
        all_dirs = [d for d in all_dirs if f"sdpa_debug_{args.call_id:04d}" in d]
        if not all_dirs:
            print(f"No call found with ID {args.call_id}")
            return

    # Verify each call
    results = []
    for call_dir in all_dirs:
        # Check if we should verify this kernel
        info = load_call_info(call_dir)
        if args.kernel != "all" and info["kernel_name"] != args.kernel:
            continue

        try:
            passed = verify_kernel_call(
                call_dir, rtol=args.rtol, atol=args.atol, device=args.device
            )
            results.append((call_dir, info["kernel_name"], info["call_id"], passed))
        except Exception as e:
            print(f"\nERROR verifying {call_dir}: {e}")
            import traceback

            traceback.print_exc()
            results.append(
                (
                    call_dir,
                    info.get("kernel_name", "unknown"),
                    info.get("call_id", -1),
                    False,
                )
            )

    # Print summary
    print(f"\n{'='*80}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total calls verified: {len(results)}")

    passed = sum(1 for r in results if r[3])
    failed = len(results) - passed

    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"\nDetailed Results:")

    for call_dir, kernel_name, call_id, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Call {call_id:04d} ({kernel_name}): {status}")

    print(f"{'='*80}\n")

    # Exit with appropriate code
    if failed > 0:
        exit(1)
    else:
        print("All verifications passed! ✓")
        exit(0)


if __name__ == "__main__":
    main()
