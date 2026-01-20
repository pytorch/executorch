#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for the MLX delegate's RoPE (Rotary Position Embedding) operation.

Tests the mlx.apply_rope custom op with various configurations:
- Static position (int literal)
- Dynamic position (via tensor.item())
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

# Import custom ops to register mlx.apply_rope
from executorch.backends.apple.mlx import ops  # noqa: F401
from executorch.backends.apple.mlx.test.test_utils import (
    OpTestCase,
    print_mlx_graph_summary,
    rebuild_op_test_runner,
    run_cpp_test_runner,
)


class RopeModel(nn.Module):
    """
    Model that applies RoPE with dynamic position via tensor.item().

    This tests the dynamic position code path where pos is a SymInt.
    """

    def __init__(
        self,
        head_dim: int = 64,
        traditional: bool = False,
        base: float = 500000.0,
        scale: float = 1.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.traditional = traditional
        self.base = base
        self.scale = scale

    def forward(
        self,
        q: torch.Tensor,  # [B, Hq, T, D]
        k: torch.Tensor,  # [B, Hk, T, D]
        pos_tensor: torch.Tensor,  # Scalar tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to q and k with dynamic position."""
        # Use .item() to get SymInt from scalar tensor
        pos = pos_tensor.item()

        q_rot, k_rot = torch.ops.mlx.apply_rope(
            q,
            k,
            self.head_dim,
            pos,
            self.traditional,
            self.base,
            self.scale,
            None,  # freqs
        )
        return q_rot, k_rot


class RopeTest(OpTestCase):
    """Test case for RoPE with dynamic position."""

    name = "rope"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        batch_size: int = 1,
        num_heads_q: int = 8,
        num_heads_k: int = 8,
        seq_len: int = 16,
        head_dim: int = 64,
        pos: int = 0,
        traditional: bool = False,
        base: float = 500000.0,
        scale: float = 1.0,
    ):
        self.batch_size = batch_size
        self.num_heads_q = num_heads_q
        self.num_heads_k = num_heads_k
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.pos = pos
        self.traditional = traditional
        self.base = base
        self.scale = scale
        self.name = "rope"

    def create_model(self) -> nn.Module:
        return RopeModel(
            head_dim=self.head_dim,
            traditional=self.traditional,
            base=self.base,
            scale=self.scale,
        )

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        q = torch.randn(self.batch_size, self.num_heads_q, self.seq_len, self.head_dim)
        k = torch.randn(self.batch_size, self.num_heads_k, self.seq_len, self.head_dim)
        pos_tensor = torch.tensor(self.pos, dtype=torch.int64)
        return (q, k, pos_tensor)

    def get_dynamic_shapes(self) -> Optional[Dict]:
        return None


class RopeTestDynamicPos(OpTestCase):
    """Test case for RoPE with dynamic position."""

    name = "rope_dynamic_pos"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        batch_size: int = 1,
        num_heads_q: int = 8,
        num_heads_k: int = 8,
        seq_len: int = 16,
        head_dim: int = 64,
        export_pos: int = 0,
        test_pos: int = 32,
        traditional: bool = False,
        base: float = 500000.0,
        scale: float = 1.0,
    ):
        self.batch_size = batch_size
        self.num_heads_q = num_heads_q
        self.num_heads_k = num_heads_k
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.export_pos = export_pos
        self.test_pos = test_pos
        self.traditional = traditional
        self.base = base
        self.scale = scale
        self.name = "rope_dynamic_pos"

    def create_model(self) -> nn.Module:
        return RopeModel(
            head_dim=self.head_dim,
            traditional=self.traditional,
            base=self.base,
            scale=self.scale,
        )

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        """Create inputs for export."""
        q = torch.randn(self.batch_size, self.num_heads_q, self.seq_len, self.head_dim)
        k = torch.randn(self.batch_size, self.num_heads_k, self.seq_len, self.head_dim)
        pos_tensor = torch.tensor(self.export_pos, dtype=torch.int64)
        return (q, k, pos_tensor)

    def create_test_inputs(self) -> Tuple[torch.Tensor, ...]:
        """Create inputs for testing with different position."""
        q = torch.randn(self.batch_size, self.num_heads_q, self.seq_len, self.head_dim)
        k = torch.randn(self.batch_size, self.num_heads_k, self.seq_len, self.head_dim)
        pos_tensor = torch.tensor(self.test_pos, dtype=torch.int64)
        return (q, k, pos_tensor)

    def get_dynamic_shapes(self) -> Optional[Dict]:
        return None


def run_rope_test(test: OpTestCase, verbose: bool = False) -> bool:
    """Run a RoPE test."""
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
    parser = argparse.ArgumentParser(description="Test MLX RoPE operation")
    parser.add_argument(
        "action",
        choices=["generate", "compare", "run"],
        help="Action to perform: generate test files, compare outputs, or run full test",
    )
    parser.add_argument(
        "--test",
        choices=["static", "dynamic_pos", "all"],
        default="all",
        help="Which test to run: static (static position), dynamic_pos (dynamic position), or all (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1)",
    )
    parser.add_argument(
        "--num-heads-q",
        type=int,
        default=8,
        help="Number of query heads (default: 8)",
    )
    parser.add_argument(
        "--num-heads-k",
        type=int,
        default=8,
        help="Number of key heads (default: 8)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=16,
        help="Sequence length (default: 16)",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=64,
        help="Head dimension (default: 64)",
    )
    parser.add_argument(
        "--pos",
        type=int,
        default=0,
        help="Position for static test (default: 0)",
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
            RopeTest(
                batch_size=args.batch_size,
                num_heads_q=args.num_heads_q,
                num_heads_k=args.num_heads_k,
                seq_len=args.seq_len,
                head_dim=args.head_dim,
                pos=args.pos,
            )
        )
    if args.test in ["dynamic_pos", "all"]:
        tests.append(
            RopeTestDynamicPos(
                batch_size=args.batch_size,
                num_heads_q=args.num_heads_q,
                num_heads_k=args.num_heads_k,
                seq_len=args.seq_len,
                head_dim=args.head_dim,
                export_pos=0,
                test_pos=32,
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
            passed = run_rope_test(test, verbose=args.verbose)
            if not passed:
                all_passed = False

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
