#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for permute and transpose ops using the MLX delegate.

These operations are fundamental for reshaping tensors, especially for
converting between different tensor layouts like [B, S, H, D] <-> [B, H, S, D].

Usage:
    # Run via run_all_tests (recommended):
    python -m executorch.backends.apple.mlx.test.run_all_tests permute

    # Run directly:
    python -m executorch.backends.apple.mlx.test.test_permute run
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test, run_op_test_main


class PermuteModel(nn.Module):
    """Model that permutes tensor dimensions."""

    def __init__(self, dims: Tuple[int, ...] = (0, 2, 1, 3)):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self.dims)


class TransposeModel(nn.Module):
    """Model that transposes two dimensions."""

    def __init__(self, dim0: int = 1, dim1: int = 2):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim0, self.dim1)


@register_test
class PermuteTest(OpTestCase):
    """Test case for permute and transpose ops."""

    name = "permute"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        shape: Tuple[int, ...] = (2, 8, 16, 64),
        variant: str = "permute",
        permute_dims: Tuple[int, ...] = (0, 2, 1, 3),
        transpose_dims: Tuple[int, int] = (1, 2),
    ):
        self.shape = shape
        self.variant = variant
        self.permute_dims = permute_dims
        self.transpose_dims = transpose_dims

        # Build unique test name
        if variant == "transpose":
            self.name = "transpose"
        else:
            self.name = "permute"

    @classmethod
    def get_test_configs(cls) -> List["PermuteTest"]:
        """Return all test configurations to run."""
        return [
            # Permute: [B, H, S, D] -> [B, S, H, D] (common for attention)
            cls(variant="permute", permute_dims=(0, 2, 1, 3)),
            # Transpose: swap dims 1 and 2
            cls(variant="transpose", transpose_dims=(1, 2)),
        ]

    def create_model(self) -> nn.Module:
        if self.variant == "transpose":
            return TransposeModel(self.transpose_dims[0], self.transpose_dims[1])
        else:
            return PermuteModel(self.permute_dims)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        return (x,)

    def get_dynamic_shapes(self):
        return None


# Factory for CLI usage
def _create_from_args(args) -> PermuteTest:
    if args is None:
        return PermuteTest()
    shape = tuple(int(x) for x in args.shape.split(","))
    return PermuteTest(
        shape=shape,
        variant=args.variant,
    )


def _add_args(parser):
    parser.add_argument(
        "--shape",
        type=str,
        default="2,8,16,64",
        help="Tensor shape (default: 2,8,16,64)",
    )
    parser.add_argument(
        "--variant",
        choices=["permute", "transpose"],
        default="permute",
        help="Which variant to test",
    )


if __name__ == "__main__":
    run_op_test_main(
        _create_from_args, "Test permute/transpose on MLX delegate", _add_args
    )
