#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for tensor slicing using the MLX delegate.

This tests aten.slice.Tensor which extracts a portion of a tensor.

Usage:
    # Run via run_all_tests (recommended):
    python -m executorch.backends.apple.mlx.test.run_all_tests slice

    # Run directly:
    python -m executorch.backends.apple.mlx.test.test_slice run
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test, run_op_test_main


class SliceModel(nn.Module):
    """Model that slices a tensor along dimension 1."""

    def __init__(self, start: int, stop: int):
        super().__init__()
        self.start = start
        self.stop = stop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Slice along dimension 1
        return x[:, self.start : self.stop]


class SliceDim0Model(nn.Module):
    """Model that slices a tensor along dimension 0."""

    def __init__(self, start: int, stop: int):
        super().__init__()
        self.start = start
        self.stop = stop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[self.start : self.stop]


@register_test
class SliceTest(OpTestCase):
    """Test case for tensor slicing."""

    name = "slice"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 16, 8),
        dim: int = 1,
        start: int = 2,
        stop: int = 10,
    ):
        self.shape = shape
        self.dim = dim
        self.start = start
        self.stop = stop
        self.name = f"slice_dim{dim}_{start}to{stop}"

    @classmethod
    def get_test_configs(cls) -> List["SliceTest"]:
        """Return all test configurations to run."""
        return [
            cls(shape=(4, 16, 8), dim=1, start=2, stop=10),  # slice middle dim
            cls(shape=(8, 8), dim=0, start=1, stop=5),  # slice first dim
            cls(shape=(2, 32, 4), dim=1, start=0, stop=16),  # slice from start
        ]

    def create_model(self) -> nn.Module:
        if self.dim == 0:
            return SliceDim0Model(self.start, self.stop)
        return SliceModel(self.start, self.stop)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        return (x,)

    def get_dynamic_shapes(self):
        return None


# Factory for CLI usage
def _create_from_args(args) -> SliceTest:
    if args is None:
        return SliceTest()
    return SliceTest(
        shape=tuple(args.shape),
        dim=args.dim,
        start=args.start,
        stop=args.stop,
    )


def _add_args(parser):
    parser.add_argument(
        "--shape", type=int, nargs="+", default=[4, 16, 8], help="Input shape"
    )
    parser.add_argument("--dim", type=int, default=1, help="Dimension to slice")
    parser.add_argument("--start", type=int, default=2, help="Slice start index")
    parser.add_argument("--stop", type=int, default=10, help="Slice stop index")


if __name__ == "__main__":
    run_op_test_main(
        _create_from_args, "Test tensor slicing on MLX delegate", _add_args
    )
