#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for unsqueeze op using the MLX delegate.

Unsqueeze adds a dimension of size 1 at a specified position.
It's commonly used for broadcasting and dimension manipulation.

Usage:
    # Run via run_all_tests (recommended):
    python -m executorch.backends.apple.mlx.test.run_all_tests unsqueeze

    # Run directly:
    python -m executorch.backends.apple.mlx.test.test_unsqueeze run
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test, run_op_test_main


class UnsqueezeModel(nn.Module):
    """Model that unsqueezes a tensor at a given dimension."""

    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(self.dim)


@register_test
class UnsqueezeTest(OpTestCase):
    """Test case for unsqueeze op."""

    name = "unsqueeze"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        shape: Tuple[int, ...] = (2, 16, 64),
        dim: int = 0,
    ):
        self.shape = shape
        self.dim = dim

        # Build unique test name
        self.name = f"unsqueeze_dim{dim}"

    @classmethod
    def get_test_configs(cls) -> List["UnsqueezeTest"]:
        """Return all test configurations to run."""
        return [
            cls(dim=0),  # unsqueeze at beginning
            cls(dim=1),  # unsqueeze in middle
            cls(dim=-1),  # unsqueeze at end (negative index)
        ]

    def create_model(self) -> nn.Module:
        return UnsqueezeModel(self.dim)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        return (x,)

    def get_dynamic_shapes(self):
        return None


# Factory for CLI usage
def _create_from_args(args) -> UnsqueezeTest:
    if args is None:
        return UnsqueezeTest()
    shape = tuple(int(x) for x in args.shape.split(","))
    return UnsqueezeTest(shape=shape, dim=args.dim)


def _add_args(parser):
    parser.add_argument(
        "--shape", type=str, default="2,16,64", help="Tensor shape (default: 2,16,64)"
    )
    parser.add_argument(
        "--dim", type=int, default=0, help="Dimension to unsqueeze at (default: 0)"
    )


if __name__ == "__main__":
    run_op_test_main(_create_from_args, "Test unsqueeze on MLX delegate", _add_args)
