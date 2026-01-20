#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for mul op using the MLX delegate.

Usage:
    # Run tensor * tensor test:
    python -m executorch.backends.apple.mlx.test.test_mul run

    # Run tensor * scalar test:
    python -m executorch.backends.apple.mlx.test.test_mul run --scalar 2.5

    # Test with dynamic batch dimension:
    python -m executorch.backends.apple.mlx.test.test_mul run --dynamic-batch

    # Tensor * scalar with dynamic batch:
    python -m executorch.backends.apple.mlx.test.test_mul run --scalar 2.5 --dynamic-batch
"""

import argparse
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.export import Dim

from .test_utils import OpTestCase, print_mlx_graph_summary, rebuild_op_test_runner


class MulTensorModel(nn.Module):
    """Multiply two tensors."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * y


class MulScalarModel(nn.Module):
    """Multiply tensor and scalar."""

    def __init__(self, scalar: float = 1.0):
        super().__init__()
        self.scalar = scalar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scalar


class MulTest(OpTestCase):
    """Test case for mul op."""

    name = "mul"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        shape: Tuple[int, ...] = (2, 16, 64),
        scalar: Optional[float] = None,
        dynamic_batch: bool = False,
        test_batch_size: Optional[int] = None,
    ):
        self.shape = shape
        self.scalar = scalar  # If set, test tensor * scalar instead of tensor * tensor
        self.dynamic_batch = dynamic_batch
        self.test_batch_size = (
            test_batch_size if test_batch_size is not None else shape[0]
        )

        # Build the name
        name_parts = ["mul"]
        if scalar is not None:
            name_parts.append("scalar")
        if dynamic_batch:
            name_parts.append("dynamic_batch")
        self.name = "_".join(name_parts)

    def create_model(self) -> nn.Module:
        if self.scalar is not None:
            return MulScalarModel(self.scalar)
        else:
            return MulTensorModel()

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        """Create inputs for export (tracing)."""
        x = torch.randn(self.shape)
        if self.scalar is not None:
            return (x,)
        else:
            y = torch.randn(self.shape)
            return (x, y)

    def create_test_inputs(self) -> Tuple[torch.Tensor, ...]:
        """Create inputs for testing - may use different batch size for dynamic tests."""
        test_shape = (self.test_batch_size,) + self.shape[1:]
        x = torch.randn(test_shape)
        if self.scalar is not None:
            return (x,)
        else:
            y = torch.randn(test_shape)
            return (x, y)

    def get_dynamic_shapes(self) -> Optional[Dict]:
        """Return dynamic shapes specification for torch.export."""
        if not self.dynamic_batch:
            return None

        batch_dim = Dim("batch", min=1, max=32)
        if self.scalar is not None:
            return {"x": {0: batch_dim}}
        else:
            return {"x": {0: batch_dim}, "y": {0: batch_dim}}


def main():
    parser = argparse.ArgumentParser(description="Test mul op on MLX delegate")
    parser.add_argument(
        "action",
        choices=["generate", "compare", "run"],
        help="Action to perform",
    )
    parser.add_argument(
        "--shape",
        type=str,
        default="2,16,64",
        help="Tensor shape as comma-separated values (default: 2,16,64)",
    )
    parser.add_argument(
        "--scalar",
        type=float,
        default=None,
        help="If provided, test tensor * scalar instead of tensor * tensor",
    )
    parser.add_argument(
        "--dynamic-batch", action="store_true", help="Test with dynamic batch dimension"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=None,
        help="Batch size for testing (default: same as shape[0])",
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

    # Parse shape
    shape = tuple(int(x) for x in args.shape.split(","))

    # Create test case
    test = MulTest(
        shape=shape,
        scalar=args.scalar,
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
