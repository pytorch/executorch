#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for reshape/view op using the MLX delegate.

Usage:
    # Generate test files:
    python -m executorch.backends.apple.mlx.test.test_reshape generate

    # Compare outputs after running C++ binary:
    python -m executorch.backends.apple.mlx.test.test_reshape compare

    # Run full test (generate + run C++ + compare):
    python -m executorch.backends.apple.mlx.test.test_reshape run

    # Test with dynamic batch dimension:
    python -m executorch.backends.apple.mlx.test.test_reshape run --dynamic-batch
"""

import argparse
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.export import Dim

from .test_utils import OpTestCase, print_mlx_graph_summary, rebuild_op_test_runner


class ReshapeModel(nn.Module):
    """Simple reshape model for testing."""

    def __init__(self, target_shape: Tuple[int, ...]):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*self.target_shape)


class ReshapeDynamicBatchModel(nn.Module):
    """Reshape model that preserves batch dimension."""

    def __init__(self, inner_shape: Tuple[int, ...]):
        super().__init__()
        self.inner_shape = inner_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        return x.view(B, *self.inner_shape)


class ReshapeTest(OpTestCase):
    """Test case for reshape/view."""

    name = "reshape"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        input_shape: Tuple[int, ...] = (2, 3, 4),
        target_shape: Tuple[int, ...] = (2, 12),
        dynamic_batch: bool = False,
        test_batch_size: Optional[int] = None,
    ):
        self.input_shape = input_shape
        self.target_shape = target_shape
        self.dynamic_batch = dynamic_batch
        self.test_batch_size = (
            test_batch_size if test_batch_size is not None else input_shape[0]
        )

        # Build the name
        name_parts = ["reshape"]
        if dynamic_batch:
            name_parts.append("dynamic_batch")
        self.name = "_".join(name_parts)

    def create_model(self) -> nn.Module:
        if self.dynamic_batch:
            # For dynamic batch, target_shape should not include batch dim
            inner_shape = self.target_shape[1:]  # Remove batch dim
            return ReshapeDynamicBatchModel(inner_shape)
        else:
            return ReshapeModel(self.target_shape)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        """Create inputs for export (tracing)."""
        x = torch.randn(*self.input_shape)
        return (x,)

    def create_test_inputs(self) -> Tuple[torch.Tensor, ...]:
        """Create inputs for testing - may use different batch size for dynamic tests."""
        if self.dynamic_batch:
            # Replace batch dimension with test_batch_size
            test_shape = (self.test_batch_size,) + self.input_shape[1:]
            x = torch.randn(*test_shape)
        else:
            x = torch.randn(*self.input_shape)
        return (x,)

    def get_dynamic_shapes(self) -> Optional[Dict]:
        """Return dynamic shapes specification for torch.export."""
        if not self.dynamic_batch:
            return None

        batch_dim = Dim("batch", min=1, max=32)
        return {"x": {0: batch_dim}}


def main():
    parser = argparse.ArgumentParser(description="Test reshape/view op on MLX delegate")
    parser.add_argument(
        "action",
        choices=["generate", "compare", "run"],
        help="Action to perform: generate (create test files), compare (compare outputs), run (full test)",
    )
    parser.add_argument(
        "--input-shape",
        type=str,
        default="2,3,4",
        help="Input shape as comma-separated ints (default: 2,3,4)",
    )
    parser.add_argument(
        "--target-shape",
        type=str,
        default="2,12",
        help="Target shape as comma-separated ints (default: 2,12)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=None,
        help="Batch size for testing (default: same as input). Use with --dynamic-batch to test different batch sizes.",
    )
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

    # Parse shapes
    input_shape = tuple(int(x) for x in args.input_shape.split(","))
    target_shape = tuple(int(x) for x in args.target_shape.split(","))

    # Rebuild if requested
    if args.rebuild:
        if not rebuild_op_test_runner(verbose=args.verbose):
            sys.exit(1)

    # Create test case
    test = ReshapeTest(
        input_shape=input_shape,
        target_shape=target_shape,
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
