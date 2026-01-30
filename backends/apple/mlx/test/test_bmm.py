#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for bmm (batch matrix multiplication) op using the MLX delegate.

Usage:
    # Run via run_all_tests (recommended):
    python -m executorch.backends.apple.mlx.test.run_all_tests bmm

    # Run directly:
    python -m executorch.backends.apple.mlx.test.test_bmm run
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test, run_op_test_main


class BmmModel(nn.Module):
    """Model that performs batch matrix multiplication."""

    def __init__(self, batch_size: int, n: int, m: int, p: int):
        super().__init__()
        # Create constant weight tensor for testing
        self.weight = nn.Parameter(torch.randn(batch_size, m, p))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # bmm(x, weight) where x is [B, N, M], weight is [B, M, P]
        # Result is [B, N, P]
        return torch.bmm(x, self.weight)


@register_test
class BmmTest(OpTestCase):
    """Test case for bmm (batch matrix multiplication).

    bmm performs batched matrix multiplication on 3D tensors.
    """

    name = "bmm"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        batch_size: int = 4,
        n: int = 8,
        m: int = 16,
        p: int = 32,
    ):
        self.batch_size = batch_size
        self.n = n
        self.m = m
        self.p = p
        self.name = f"bmm_{batch_size}x{n}x{m}x{p}"

    @classmethod
    def get_test_configs(cls) -> List["BmmTest"]:
        """Return all test configurations to run."""
        return [
            cls(batch_size=4, n=8, m=16, p=32),
            cls(batch_size=2, n=64, m=64, p=32),
        ]

    def create_model(self) -> nn.Module:
        return BmmModel(self.batch_size, self.n, self.m, self.p)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.batch_size, self.n, self.m)
        return (x,)

    def get_dynamic_shapes(self):
        return None


# Factory for CLI usage
def _create_from_args(args) -> BmmTest:
    if args is None:
        return BmmTest()
    return BmmTest(
        batch_size=args.batch_size,
        n=args.n,
        m=args.m,
        p=args.p,
    )


def _add_args(parser):
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--n", type=int, default=8, help="N dimension")
    parser.add_argument("--m", type=int, default=16, help="M dimension")
    parser.add_argument("--p", type=int, default=32, help="P dimension")


if __name__ == "__main__":
    run_op_test_main(_create_from_args, "Test bmm on MLX delegate", _add_args)
