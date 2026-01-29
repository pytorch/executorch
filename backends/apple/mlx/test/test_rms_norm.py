#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for RMSNorm op using the MLX delegate.

This tests the custom mlx::rms_norm op which is commonly used in
LLaMA-style models for normalization.

Usage:
    # Run via run_all_tests (recommended):
    python -m executorch.backends.apple.mlx.test.run_all_tests rms_norm

    # Run directly with custom args:
    python -m executorch.backends.apple.mlx.test.test_rms_norm run --hidden-dim 256
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from executorch.backends.apple.mlx import custom_ops  # noqa: F401 - registers ops

from .test_utils import OpTestCase, register_test, run_op_test_main


class RMSNormModel(nn.Module):
    """Model using the custom mlx::rms_norm op."""

    def __init__(self, hidden_dim: int = 64, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.mlx.rms_norm(x, self.weight, self.eps)


@register_test
class RMSNormTest(OpTestCase):
    """Test case for mlx::rms_norm op."""

    name = "rms_norm"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        hidden_dim: int = 64,
        batch_size: int = 2,
        seq_len: int = 16,
        eps: float = 1e-5,
    ):
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.eps = eps
        self.name = "rms_norm"

    @classmethod
    def get_test_configs(cls) -> List["RMSNormTest"]:
        """Return all test configurations to run."""
        return [
            cls(),  # default config
            cls(hidden_dim=128, eps=1e-6),  # different hidden dim and eps
        ]

    def create_model(self) -> nn.Module:
        return RMSNormModel(self.hidden_dim, self.eps)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        return (x,)

    def get_dynamic_shapes(self):
        return None


# Factory for CLI usage
def _create_from_args(args) -> RMSNormTest:
    if args is None:
        return RMSNormTest()
    return RMSNormTest(
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        eps=args.eps,
    )


def _add_args(parser):
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length")
    parser.add_argument("--eps", type=float, default=1e-5, help="Epsilon for RMSNorm")


if __name__ == "__main__":
    run_op_test_main(_create_from_args, "Test mlx::rms_norm on MLX delegate", _add_args)
