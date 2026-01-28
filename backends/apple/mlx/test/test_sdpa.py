#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for scaled_dot_product_attention (SDPA) using the MLX delegate.

Usage:
    # Run via run_all_tests (recommended):
    python -m executorch.backends.apple.mlx.test.run_all_tests sdpa

    # Run directly with custom args:
    python -m executorch.backends.apple.mlx.test.test_sdpa run --gqa --num-kv-heads 4
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .test_utils import OpTestCase, register_test, run_op_test_main


class SDPAModel(nn.Module):
    """Basic scaled dot product attention."""

    def __init__(self, is_causal: bool = False):
        super().__init__()
        self.is_causal = is_causal

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)


class SDPAWithMaskModel(nn.Module):
    """SDPA with explicit attention mask."""

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class GQAModel(nn.Module):
    """Grouped Query Attention - fewer KV heads than Q heads."""

    def __init__(self, num_heads: int, num_kv_heads: int, is_causal: bool = False):
        super().__init__()
        self.num_groups = num_heads // num_kv_heads
        self.is_causal = is_causal

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)
        return F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)


@register_test
class SDPATest(OpTestCase):
    """Test case for SDPA."""

    name = "sdpa"
    rtol = 1e-3
    atol = 1e-3

    def __init__(
        self,
        batch_size: int = 2,
        num_heads: int = 8,
        seq_len: int = 32,
        head_dim: int = 64,
        num_kv_heads: Optional[int] = None,
        is_causal: bool = False,
        use_mask: bool = False,
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.is_causal = is_causal
        self.use_mask = use_mask

        # Build unique test name
        parts = ["sdpa"]
        if num_kv_heads is not None:
            parts.append(f"gqa{num_kv_heads}")
        if is_causal:
            parts.append("causal")
        if use_mask:
            parts.append("mask")
        self.name = "_".join(parts)

    @classmethod
    def get_test_configs(cls) -> List["SDPATest"]:
        """Return all test configurations to run."""
        return [
            cls(),  # basic
            cls(is_causal=True),  # causal
            cls(num_kv_heads=4),  # GQA
            cls(use_mask=True),  # explicit mask
        ]

    def create_model(self) -> nn.Module:
        if self.use_mask:
            return SDPAWithMaskModel()
        elif self.num_kv_heads is not None:
            return GQAModel(self.num_heads, self.num_kv_heads, self.is_causal)
        else:
            return SDPAModel(self.is_causal)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        q = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        kv_heads = self.num_kv_heads if self.num_kv_heads else self.num_heads
        k = torch.randn(self.batch_size, kv_heads, self.seq_len, self.head_dim)
        v = torch.randn(self.batch_size, kv_heads, self.seq_len, self.head_dim)

        if self.use_mask:
            mask = torch.zeros(self.batch_size, 1, self.seq_len, self.seq_len)
            mask[:, :, :, : self.seq_len // 4] = float("-inf")
            return (q, k, v, mask)
        return (q, k, v)

    def get_dynamic_shapes(self):
        return None


# Factory for CLI usage
def _create_from_args(args) -> SDPATest:
    if args is None:
        return SDPATest()
    num_kv_heads = args.num_kv_heads if getattr(args, "gqa", False) else None
    return SDPATest(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        num_kv_heads=num_kv_heads,
        is_causal=getattr(args, "causal", False),
        use_mask=getattr(args, "mask", False),
    )


def _add_args(parser):
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of Q heads")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--gqa", action="store_true", help="Use GQA")
    parser.add_argument("--num-kv-heads", type=int, default=4, help="KV heads for GQA")
    parser.add_argument("--causal", action="store_true", help="Use causal attention")
    parser.add_argument("--mask", action="store_true", help="Use explicit mask")


if __name__ == "__main__":
    run_op_test_main(_create_from_args, "Test SDPA on MLX delegate", _add_args)
