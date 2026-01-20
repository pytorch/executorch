#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for the MLX delegate's slice operation.

Tests aten.slice.Tensor with various configurations:
- Static slice (fixed start/end)
- Dynamic slice with Python syntax x[:, :, 0:end, :] where end is size-backed

NOTE on dynamic slice end values:
The `end` value in a dynamic slice MUST come from tensor.size() (a "backed" symbol),
NOT from tensor.item() (an "unbacked" symbol). This is because ExecuTorch's
spec_prop_pass.py (in to_executorch()) needs to compute dim_order from tensor strides.
When the output has a symbolic dimension from an unbacked symbol (e.g., u0 from item()),
the stride comparison fails with:
    GuardOnDataDependentSymNode: Could not guard on data-dependent expression 64*u0 < 64

Backed symbols (s64 from tensor.size()) can be reasoned about symbolically because
PyTorch knows they relate to an input tensor's dimension, while unbacked symbols
are purely data-dependent with no relationship to any tensor shape.

Example - WORKS:
    end = k_val.size(2)  # s64 - backed symbol from shape
    result = cache[:, :, 0:end, :]

Example - FAILS in spec_prop_pass:
    end = end_tensor.item()  # u0 - unbacked symbol from data
    result = cache[:, :, 0:end, :]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from executorch.backends.apple.mlx.test.test_utils import (
    OpTestCase,
    print_mlx_graph_summary,
    rebuild_op_test_runner,
    run_cpp_test_runner,
)
from torch.export import Dim


class SliceModel(nn.Module):
    """
    Simple model that slices a tensor with static indices.

    Example: x[:, :, 0:8, :] for a 4D tensor.
    """

    def __init__(
        self,
        dim: int = 2,
        start: int = 0,
        end: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.start = start
        self.end = end

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Slice the tensor on the specified dimension."""
        # x[:, :, start:end, :] for dim=2
        return x[:, :, self.start : self.end, :]


class SliceModelDynamicNarrow(nn.Module):
    """
    Model that slices using narrow with shape-based length.

    This mirrors how ExecutorTorch's KV cache works in examples/models/llama/attention.py:
        seq_length = k_val.size(dim_to_slice)
        narrowed_k = self.k_cache.narrow(dim_to_slice, start_pos, seq_length)

    The key is that seq_length comes from tensor.size() (shape-based, not data-dependent).
    """

    def __init__(
        self,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

    def forward(
        self,
        cache: torch.Tensor,  # [B, H, max_seq_len, D]
        k_val: torch.Tensor,  # [B, H, seq_len, D] - seq_len is dynamic
        start_pos: torch.Tensor,  # scalar tensor
    ) -> torch.Tensor:
        """Slice cache using narrow with shape-based length."""
        # Get start position from tensor
        pos = start_pos.item()
        torch._check_is_size(pos)
        torch._check(pos < self.max_seq_len)

        # Get length from k_val's shape (NOT data-dependent!)
        seq_length = k_val.size(2)

        # Narrow the cache: cache[:, :, pos:pos+seq_length, :]
        return cache.narrow(2, pos, seq_length)


class SliceModelDynamicEnd(nn.Module):
    """
    Model that slices with dynamic end position using Python slice syntax.

    The end value comes from tensor.size() which produces a BACKED symbol.
    This is required because ExecuTorch's spec_prop_pass.py needs to compute
    dim_order from tensor strides, which fails with unbacked symbols from item().

    Pattern: cache[:, :, 0:end, :] where end = k_val.size(2)
    """

    def __init__(
        self,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

    def forward(
        self,
        cache: torch.Tensor,  # [B, H, max_seq_len, D]
        k_val: torch.Tensor,  # [B, H, seq_len, D] - seq_len is dynamic
    ) -> torch.Tensor:
        """Slice using Python indexing with size-backed end.

        NOTE: end MUST come from tensor.size() (backed symbol), NOT tensor.item()
        (unbacked symbol). See module docstring for details.
        """
        # end comes from tensor.size() - this produces a BACKED symbol (e.g., s64)
        # that ExecuTorch's spec_prop_pass can reason about symbolically
        end = k_val.size(2)
        return cache[:, :, 0:end, :]


class SliceTest(OpTestCase):
    """Test case for slice with static indices."""

    name = "slice"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        batch_size: int = 1,
        num_heads: int = 8,
        seq_len: int = 32,
        head_dim: int = 64,
        dim: int = 2,
        start: int = 0,
        end: int = 16,
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.dim = dim
        self.start = start
        self.end = end
        self.name = "slice"

    def create_model(self) -> nn.Module:
        return SliceModel(
            dim=self.dim,
            start=self.start,
            end=self.end,
        )

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        return (x,)

    def get_dynamic_shapes(self) -> Optional[Dict]:
        return None


class SliceTestDynamicNarrow(OpTestCase):
    """Test case for slice with narrow and shape-based dynamic length."""

    name = "slice_dynamic_narrow"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        batch_size: int = 1,
        num_heads: int = 8,
        max_seq_len: int = 128,
        head_dim: int = 64,
        export_seq_len: int = 8,
        test_seq_len: int = 16,
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.export_seq_len = export_seq_len
        self.test_seq_len = test_seq_len
        self.name = "slice_dynamic_narrow"

    def create_model(self) -> nn.Module:
        return SliceModelDynamicNarrow(max_seq_len=self.max_seq_len)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        """Create inputs for export."""
        cache = torch.randn(
            self.batch_size, self.num_heads, self.max_seq_len, self.head_dim
        )
        k_val = torch.randn(
            self.batch_size, self.num_heads, self.export_seq_len, self.head_dim
        )
        start_pos = torch.tensor(0, dtype=torch.int64)
        return (cache, k_val, start_pos)

    def create_test_inputs(self) -> Tuple[torch.Tensor, ...]:
        """Create inputs for testing with different seq_len."""
        cache = torch.randn(
            self.batch_size, self.num_heads, self.max_seq_len, self.head_dim
        )
        k_val = torch.randn(
            self.batch_size, self.num_heads, self.test_seq_len, self.head_dim
        )
        start_pos = torch.tensor(4, dtype=torch.int64)  # Different start position
        return (cache, k_val, start_pos)

    def get_dynamic_shapes(self) -> Optional[Dict]:
        """k_val has dynamic seq_len dimension."""
        seq_len_dim = Dim("seq_len", min=1, max=self.max_seq_len)
        return {
            # cache: static
            "cache": None,
            # k_val: dynamic on dim 2
            "k_val": {2: seq_len_dim},
            # start_pos: static scalar
            "start_pos": None,
        }


class SliceTestDynamicEnd(OpTestCase):
    """Test case for slice with dynamic end using Python slice syntax.

    The end value comes from k_val.size(2), which is a backed symbol.
    """

    name = "slice_dynamic_end"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        batch_size: int = 1,
        num_heads: int = 8,
        max_seq_len: int = 128,
        head_dim: int = 64,
        export_seq_len: int = 8,
        test_seq_len: int = 16,
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.export_seq_len = export_seq_len
        self.test_seq_len = test_seq_len
        self.name = "slice_dynamic_end"

    def create_model(self) -> nn.Module:
        return SliceModelDynamicEnd(max_seq_len=self.max_seq_len)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        """Create inputs for export."""
        cache = torch.randn(
            self.batch_size, self.num_heads, self.max_seq_len, self.head_dim
        )
        # k_val provides the dynamic end via k_val.size(2)
        k_val = torch.randn(
            self.batch_size, self.num_heads, self.export_seq_len, self.head_dim
        )
        return (cache, k_val)

    def create_test_inputs(self) -> Tuple[torch.Tensor, ...]:
        """Create inputs for testing with different seq_len (end value)."""
        cache = torch.randn(
            self.batch_size, self.num_heads, self.max_seq_len, self.head_dim
        )
        k_val = torch.randn(
            self.batch_size, self.num_heads, self.test_seq_len, self.head_dim
        )
        return (cache, k_val)

    def get_dynamic_shapes(self) -> Optional[Dict]:
        """k_val has dynamic seq_len dimension which determines the slice end."""
        seq_len_dim = Dim("seq_len", min=1, max=self.max_seq_len)
        return {
            "cache": None,
            "k_val": {2: seq_len_dim},
        }


def run_slice_test(test: OpTestCase, verbose: bool = False) -> bool:
    """Run a slice test."""
    print(f"\n{'='*60}")
    print(f"Running test: {test.name}")
    print(f"{'='*60}\n")

    # Generate test files
    print("Step 1: Generating test files...")
    pte_path, input_path, expected_path = test.generate_test_files(verbose=verbose)

    # Print MLX graph summary
    print_mlx_graph_summary(pte_path)

    # Run C++ binary
    print("Step 2: Running C++ binary...")
    actual_path = test.get_test_dir() / "actual_output.bin"
    if not run_cpp_test_runner(pte_path, input_path, actual_path, verbose=verbose):
        return False

    # Compare outputs
    print("\nStep 3: Comparing outputs...")
    passed, message = test.compare_with_actual(actual_path)

    if passed:
        print(f"✓ PASSED: {message}")
    else:
        print(f"✗ FAILED: {message}")

    return passed


def main():
    parser = argparse.ArgumentParser(description="Test MLX slice operation")
    parser.add_argument(
        "action",
        choices=["generate", "compare", "run"],
        help="Action to perform: generate test files, compare outputs, or run full test",
    )
    parser.add_argument(
        "--test",
        choices=["static", "dynamic_narrow", "dynamic_end", "all"],
        default="all",
        help="Which test to run (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of heads (default: 8)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=32,
        help="Sequence length (default: 32)",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=64,
        help="Head dimension (default: 64)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the op_test_runner binary before running tests",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Rebuild if requested
    if args.rebuild:
        if not rebuild_op_test_runner(verbose=args.verbose):
            print("Failed to rebuild op_test_runner")
            sys.exit(1)

    # Create test instances
    tests = []
    if args.test in ["static", "all"]:
        tests.append(
            SliceTest(
                batch_size=args.batch_size,
                num_heads=args.num_heads,
                seq_len=args.seq_len,
                head_dim=args.head_dim,
                dim=2,
                start=0,
                end=16,
            )
        )
    if args.test in ["dynamic_narrow", "all"]:
        tests.append(
            SliceTestDynamicNarrow(
                batch_size=args.batch_size,
                num_heads=args.num_heads,
                max_seq_len=128,
                head_dim=args.head_dim,
                export_seq_len=8,
                test_seq_len=16,
            )
        )
    if args.test in ["dynamic_end", "all"]:
        tests.append(
            SliceTestDynamicEnd(
                batch_size=args.batch_size,
                num_heads=args.num_heads,
                max_seq_len=128,
                head_dim=args.head_dim,
                export_seq_len=8,
                test_seq_len=16,
            )
        )

    all_passed = True
    for test in tests:
        if args.action == "generate":
            test.generate_test_files(verbose=args.verbose)
            print_mlx_graph_summary(test.get_test_dir() / "model.pte")

        elif args.action == "compare":
            actual_path = test.get_test_dir() / "actual_output.bin"
            passed, message = test.compare_with_actual(actual_path)
            if passed:
                print(f"✓ PASSED: {message}")
            else:
                print(f"✗ FAILED: {message}")
                all_passed = False

        elif args.action == "run":
            passed = run_slice_test(test, verbose=args.verbose)
            if not passed:
                all_passed = False

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
