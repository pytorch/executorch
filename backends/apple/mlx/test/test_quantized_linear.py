#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for TorchAO int4 quantized nn.Linear op using the MLX delegate.

Usage:
    # Generate test files:
    python -m executorch.backends.apple.mlx.test.test_quantized_linear generate

    # Compare outputs after running C++ binary:
    python -m executorch.backends.apple.mlx.test.test_quantized_linear compare

    # Run full test (generate + run C++ + compare):
    python -m executorch.backends.apple.mlx.test.test_quantized_linear run
"""

import argparse
import sys
from typing import Tuple

import torch
import torch.nn as nn
from torch.export import Dim

from .test_utils import OpTestCase, print_mlx_graph_summary, rebuild_op_test_runner


class QuantizedLinearModel(nn.Module):
    """Simple linear layer that will be quantized."""

    def __init__(
        self, in_features: int = 64, out_features: int = 128, bias: bool = True
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class QuantizedLinearTest(OpTestCase):
    """Test case for TorchAO int4 quantized nn.Linear."""

    name = "quantized_linear"
    rtol = 0.1  # Higher tolerance for quantized ops
    atol = 0.1

    def __init__(
        self,
        in_features: int = 64,
        out_features: int = 128,
        batch_size: int = 2,
        seq_len: int = 16,
        bias: bool = True,
        group_size: int = 32,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.bias = bias
        self.group_size = group_size
        self.dtype = dtype

        # Build the name
        name_parts = ["quantized_linear", f"g{group_size}"]
        if not bias:
            name_parts.append("no_bias")
        self.name = "_".join(name_parts)

    def create_model(self) -> nn.Module:
        model = QuantizedLinearModel(
            self.in_features, self.out_features, bias=self.bias
        )
        model = model.to(self.dtype)

        # Apply TorchAO int4 quantization
        try:
            from torchao.quantization.granularity import PerGroup
            from torchao.quantization.quant_api import IntxWeightOnlyConfig, quantize_

            quantize_(
                model,
                IntxWeightOnlyConfig(
                    weight_dtype=torch.int4, granularity=PerGroup(self.group_size)
                ),
            )
        except ImportError:
            raise RuntimeError("TorchAO not installed. Run: pip install torchao")

        return model

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        """Create inputs for export (tracing)."""
        # Inputs need to match model dtype for quantized linear
        x = torch.randn(
            self.batch_size, self.seq_len, self.in_features, dtype=self.dtype
        )
        return (x,)

    def create_test_inputs(self) -> Tuple[torch.Tensor, ...]:
        """Create inputs for testing."""
        return self.create_inputs()

    def get_dynamic_shapes(self):
        """Return dynamic shapes specification for torch.export."""
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Test TorchAO int4 quantized nn.Linear op on MLX delegate"
    )
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
        "--batch-size", type=int, default=2, help="Batch size (default: 2)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=16, help="Sequence length (default: 16)"
    )
    parser.add_argument("--no-bias", action="store_true", help="Test without bias")
    parser.add_argument(
        "--group-size",
        type=int,
        default=32,
        help="Quantization group size (default: 32)",
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
    test = QuantizedLinearTest(
        in_features=args.in_features,
        out_features=args.out_features,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        bias=not args.no_bias,
        group_size=args.group_size,
    )

    if args.action == "generate":
        pte_path, input_path, expected_path = test.generate_test_files(
            verbose=args.verbose
        )
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
