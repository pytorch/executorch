#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for nn.Linear op using the MLX delegate.

Usage:
    # Generate test files:
    python -m executorch.backends.apple.mlx.test.test_linear generate

    # Compare outputs after running C++ binary:
    python -m executorch.backends.apple.mlx.test.test_linear compare

    # Run full test (generate + run C++ + compare):
    python -m executorch.backends.apple.mlx.test.test_linear run

    # Test with dynamic batch dimension:
    python -m executorch.backends.apple.mlx.test.test_linear run --dynamic-batch
"""

import argparse
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.export import Dim

from .test_utils import OpTestCase, print_mlx_graph_summary, rebuild_op_test_runner


class LinearModel(nn.Module):
    """Simple linear layer for testing."""

    def __init__(
        self, in_features: int = 64, out_features: int = 128, bias: bool = True
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class LinearTest(OpTestCase):
    """Test case for nn.Linear."""

    name = "linear"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        in_features: int = 64,
        out_features: int = 128,
        batch_size: int = 2,
        seq_len: int = 16,
        bias: bool = True,
        dynamic_batch: bool = False,
        test_batch_size: Optional[int] = None,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.bias = bias
        self.dynamic_batch = dynamic_batch
        self.test_batch_size = (
            test_batch_size if test_batch_size is not None else batch_size
        )

        # Build the name
        name_parts = ["linear"]
        if not bias:
            name_parts.append("no_bias")
        if dynamic_batch:
            name_parts.append("dynamic_batch")
        self.name = "_".join(name_parts)

    def create_model(self) -> nn.Module:
        return LinearModel(self.in_features, self.out_features, bias=self.bias)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        """Create inputs for export (tracing)."""
        x = torch.randn(self.batch_size, self.seq_len, self.in_features)
        return (x,)

    def create_test_inputs(self) -> Tuple[torch.Tensor, ...]:
        """Create inputs for testing - may use different batch size for dynamic tests."""
        x = torch.randn(self.test_batch_size, self.seq_len, self.in_features)
        return (x,)

    def get_dynamic_shapes(self) -> Optional[Dict]:
        """Return dynamic shapes specification for torch.export."""
        if not self.dynamic_batch:
            return None

        batch_dim = Dim("batch", min=1, max=32)
        return {"x": {0: batch_dim}}


def main():
    parser = argparse.ArgumentParser(description="Test nn.Linear op on MLX delegate")
    parser.add_argument(
        "action",
        choices=["generate", "compare", "run"],
        help="Action to perform: generate (create test files), compare (compare outputs), run (full test)",
    )
    parser.add_argument(
        "--in-features", type=int, default=64, help="Input features (default: 64)"
    )
    parser.add_argument(
        "--out-features", type=int, default=128, help="Output features (default: 128)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Batch size for export (default: 2)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=None,
        help="Batch size for testing (default: same as --batch-size). Use with --dynamic-batch to test different batch sizes.",
    )
    parser.add_argument(
        "--seq-len", type=int, default=16, help="Sequence length (default: 16)"
    )
    parser.add_argument("--no-bias", action="store_true", help="Test without bias")
    parser.add_argument(
        "--dynamic-batch", action="store_true", help="Test with dynamic batch dimension"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the C++ test runner before running",
    )
    args = parser.parse_args()

    # Rebuild if requested
    if args.rebuild:
        if not rebuild_op_test_runner(verbose=args.verbose):
            sys.exit(1)

    # Create test case
    test = LinearTest(
        in_features=args.in_features,
        out_features=args.out_features,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        bias=not args.no_bias,
        dynamic_batch=args.dynamic_batch,
        test_batch_size=args.test_batch_size,
    )

    if args.action == "generate":
        pte_path, input_path, expected_path = test.generate_test_files()
        print(f"\nGenerated files:")
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


if __name__ == "__main__":
    main()
