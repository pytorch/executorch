#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for RoPE (Rotary Position Embedding) using the MLX delegate.

Usage:
    # Run via run_all_tests (recommended):
    python -m executorch.backends.apple.mlx.test.run_all_tests rope

    # Run directly with custom args:
    python -m executorch.backends.apple.mlx.test.test_rope run --pos 100
"""

from typing import List, Tuple

import torch
import torch.nn as nn

# Import custom ops to register mlx.rope
from executorch.backends.apple.mlx import ops  # noqa: F401

from .test_utils import OpTestCase, register_test, run_op_test_main


class RopeModel(nn.Module):
    """Model that applies RoPE with dynamic position."""

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
        q: torch.Tensor,
        k: torch.Tensor,
        pos_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = pos_tensor.item()
        q_rot = torch.ops.mlx.rope(
            q, self.head_dim, pos, self.traditional, self.base, self.scale, None
        )
        k_rot = torch.ops.mlx.rope(
            k, self.head_dim, pos, self.traditional, self.base, self.scale, None
        )
        return q_rot, k_rot


@register_test
class RopeTest(OpTestCase):
    """Test case for RoPE."""

    name = "rope"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        batch_size: int = 1,
        num_heads: int = 8,
        seq_len: int = 16,
        head_dim: int = 64,
        pos: int = 0,
        traditional: bool = False,
        base: float = 500000.0,
        scale: float = 1.0,
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.pos = pos
        self.traditional = traditional
        self.base = base
        self.scale = scale
        self.name = "rope"

    @classmethod
    def get_test_configs(cls) -> List["RopeTest"]:
        """Return all test configurations to run."""
        return [
            cls(),  # default (pos=0)
        ]

    def create_model(self) -> nn.Module:
        return RopeModel(
            head_dim=self.head_dim,
            traditional=self.traditional,
            base=self.base,
            scale=self.scale,
        )

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        q = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        k = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        pos_tensor = torch.tensor(self.pos, dtype=torch.int64)
        return (q, k, pos_tensor)

    def get_dynamic_shapes(self):
        return None


# Factory for CLI usage
def _create_from_args(args) -> RopeTest:
    if args is None:
        return RopeTest()
    return RopeTest(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        pos=args.pos,
    )


def _add_args(parser):
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--pos", type=int, default=0, help="Position offset")


if __name__ == "__main__":
    run_op_test_main(_create_from_args, "Test RoPE on MLX delegate", _add_args)
