#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for tensor.narrow() op using the MLX delegate.

narrow(dim, start, length) extracts a slice of the tensor using
(start, length) semantics instead of (start, stop).

This is commonly used in KV cache operations.

Usage:
    # Run via run_all_tests (recommended):
    python -m executorch.backends.apple.mlx.test.run_all_tests narrow

    # Run directly:
    python -m executorch.backends.apple.mlx.test.test_narrow run
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test, run_op_test_main


class NarrowModel(nn.Module):
    """Model that narrows a tensor along a dimension."""

    def __init__(self, dim: int, start: int, length: int):
        super().__init__()
        self.dim = dim
        self.start = start
        self.length = length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.narrow(self.dim, self.start, self.length)


@register_test
class NarrowTest(OpTestCase):
    """Test case for tensor.narrow()."""

    name = "narrow"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 16, 8),
        dim: int = 1,
        start: int = 2,
        length: int = 8,
    ):
        self.shape = shape
        self.dim = dim
        self.start = start
        self.length = length
        self.name = f"narrow_dim{dim}_start{start}_len{length}"

    @classmethod
    def get_test_configs(cls) -> List["NarrowTest"]:
        """Return all test configurations to run."""
        return [
            cls(shape=(4, 16, 8), dim=1, start=2, length=8),  # narrow middle dim
            cls(shape=(8, 8), dim=0, start=1, length=4),  # narrow first dim
            cls(shape=(2, 32, 4), dim=1, start=0, length=16),  # narrow from start
        ]

    def create_model(self) -> nn.Module:
        return NarrowModel(self.dim, self.start, self.length)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        return (x,)

    def get_dynamic_shapes(self):
        return None


# Factory for CLI usage
def _create_from_args(args) -> NarrowTest:
    if args is None:
        return NarrowTest()
    return NarrowTest(
        shape=tuple(args.shape),
        dim=args.dim,
        start=args.start,
        length=args.length,
    )


def _add_args(parser):
    parser.add_argument(
        "--shape", type=int, nargs="+", default=[4, 16, 8], help="Input shape"
    )
    parser.add_argument("--dim", type=int, default=1, help="Dimension to narrow")
    parser.add_argument("--start", type=int, default=2, help="Start index")
    parser.add_argument("--length", type=int, default=8, help="Length of slice")


if __name__ == "__main__":
    run_op_test_main(
        _create_from_args, "Test tensor.narrow() on MLX delegate", _add_args
    )
