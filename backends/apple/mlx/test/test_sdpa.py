#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for scaled_dot_product_attention (SDPA) op using the MLX delegate.

Tests cover:
1. Basic SDPA (multi-head attention)
2. Grouped Query Attention (GQA) - uses repeat_interleave pattern
3. Dynamic sequence length

Usage:
    # Run basic SDPA test:
    python -m executorch.backends.apple.mlx.test.test_sdpa run

    # Run GQA test (grouped query attention with fewer KV heads):
    python -m executorch.backends.apple.mlx.test.test_sdpa run --gqa --num-kv-heads 4

    # Test with dynamic sequence length:
    python -m executorch.backends.apple.mlx.test.test_sdpa run --dynamic-seqlen

    # GQA with dynamic seqlen:
    python -m executorch.backends.apple.mlx.test.test_sdpa run --gqa --num-kv-heads 4 --dynamic-seqlen

    # Causal attention:
    python -m executorch.backends.apple.mlx.test.test_sdpa run --causal
"""

import argparse
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import Dim

from .test_utils import OpTestCase, print_mlx_graph_summary, rebuild_op_test_runner


class SDPAModel(nn.Module):
    """Basic scaled dot product attention."""

    def __init__(self, is_causal: bool = False):
        super().__init__()
        self.is_causal = is_causal

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        # q, k, v shapes: [batch, num_heads, seq_len, head_dim]
        return F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)


class GQAModel(nn.Module):
    """
    Grouped Query Attention - uses fewer KV heads than Q heads.

    The K and V tensors have fewer heads, which are repeated to match Q heads.
    This pattern is detected by MLX and fused into a single SDPA op.
    """

    def __init__(self, num_heads: int, num_kv_heads: int, is_causal: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.is_causal = is_causal

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        # q shape: [batch, num_heads, seq_len, head_dim]
        # k, v shapes: [batch, num_kv_heads, seq_len, head_dim]

        # Repeat K and V to match Q's number of heads
        # This repeat_interleave pattern is detected and fused by MLX
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        return F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)


class SDPATest(OpTestCase):
    """Test case for SDPA op."""

    name = "sdpa"
    rtol = 1e-3  # SDPA can have larger numerical differences
    atol = 1e-3

    def __init__(
        self,
        batch_size: int = 2,
        num_heads: int = 8,
        seq_len: int = 32,
        head_dim: int = 64,
        num_kv_heads: Optional[int] = None,  # If set, use GQA
        is_causal: bool = False,
        dynamic_seqlen: bool = False,
        test_seq_len: Optional[int] = None,
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.is_causal = is_causal
        self.dynamic_seqlen = dynamic_seqlen
        self.test_seq_len = test_seq_len if test_seq_len is not None else seq_len

        # Build the name
        name_parts = ["sdpa"]
        if num_kv_heads is not None:
            name_parts.append(f"gqa{num_kv_heads}")
        if is_causal:
            name_parts.append("causal")
        if dynamic_seqlen:
            name_parts.append("dynamic_seqlen")
        self.name = "_".join(name_parts)

    def create_model(self) -> nn.Module:
        if self.num_kv_heads is not None:
            return GQAModel(
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                is_causal=self.is_causal,
            )
        else:
            return SDPAModel(is_causal=self.is_causal)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        """Create inputs for export (tracing)."""
        q = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)

        if self.num_kv_heads is not None:
            # GQA: K and V have fewer heads
            k = torch.randn(
                self.batch_size, self.num_kv_heads, self.seq_len, self.head_dim
            )
            v = torch.randn(
                self.batch_size, self.num_kv_heads, self.seq_len, self.head_dim
            )
        else:
            # Standard MHA: K and V have same heads as Q
            k = torch.randn(
                self.batch_size, self.num_heads, self.seq_len, self.head_dim
            )
            v = torch.randn(
                self.batch_size, self.num_heads, self.seq_len, self.head_dim
            )

        return (q, k, v)

    def create_test_inputs(self) -> Tuple[torch.Tensor, ...]:
        """Create inputs for testing - may use different seq_len for dynamic tests."""
        q = torch.randn(
            self.batch_size, self.num_heads, self.test_seq_len, self.head_dim
        )

        if self.num_kv_heads is not None:
            k = torch.randn(
                self.batch_size, self.num_kv_heads, self.test_seq_len, self.head_dim
            )
            v = torch.randn(
                self.batch_size, self.num_kv_heads, self.test_seq_len, self.head_dim
            )
        else:
            k = torch.randn(
                self.batch_size, self.num_heads, self.test_seq_len, self.head_dim
            )
            v = torch.randn(
                self.batch_size, self.num_heads, self.test_seq_len, self.head_dim
            )

        return (q, k, v)

    def get_dynamic_shapes(self) -> Optional[Dict]:
        """Return dynamic shapes specification for torch.export."""
        if not self.dynamic_seqlen:
            return None

        seq_dim = Dim("seq_len", min=1, max=4096)

        if self.num_kv_heads is not None:
            # GQA: Q has more heads than K, V
            return {
                "q": {2: seq_dim},
                "k": {2: seq_dim},
                "v": {2: seq_dim},
            }
        else:
            return {
                "q": {2: seq_dim},
                "k": {2: seq_dim},
                "v": {2: seq_dim},
            }


def main():
    parser = argparse.ArgumentParser(description="Test SDPA op on MLX delegate")
    parser.add_argument(
        "action",
        choices=["generate", "compare", "run"],
        help="Action to perform",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size (default: 2)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of query heads (default: 8)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=32,
        help="Sequence length for export (default: 32)",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=64,
        help="Head dimension (default: 64)",
    )
    parser.add_argument(
        "--gqa",
        action="store_true",
        help="Use Grouped Query Attention",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=None,
        help="Number of KV heads for GQA (default: num_heads // 2)",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Use causal attention mask",
    )
    parser.add_argument(
        "--dynamic-seqlen",
        action="store_true",
        help="Test with dynamic sequence length",
    )
    parser.add_argument(
        "--test-seq-len",
        type=int,
        default=None,
        help="Sequence length for testing (default: same as --seq-len)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
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

    # Determine num_kv_heads for GQA
    num_kv_heads = None
    if args.gqa:
        num_kv_heads = args.num_kv_heads if args.num_kv_heads else args.num_heads // 2

    # Create test case
    test = SDPATest(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        num_kv_heads=num_kv_heads,
        is_causal=args.causal,
        dynamic_seqlen=args.dynamic_seqlen,
        test_seq_len=args.test_seq_len,
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
