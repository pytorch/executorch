#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for softmax op using the MLX delegate.

Usage:
    # Run via run_all_tests (recommended):
    python -m executorch.backends.apple.mlx.test.run_all_tests softmax

    # Run directly with custom args:
    python -m executorch.backends.apple.mlx.test.test_softmax run --dim -1
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test, run_op_test_main


class SoftmaxModel(nn.Module):
    """Model that performs softmax along a specified dimension."""

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(x, dim=self.dim)


@register_test
class SoftmaxTest(OpTestCase):
    """Test case for softmax op."""

    name = "softmax"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        shape: Tuple[int, ...] = (2, 3, 4),
        dim: int = -1,
    ):
        self.shape = shape
        self.dim = dim

        # Build unique test name based on shape and dim
        shape_str = "x".join(str(s) for s in shape)
        self.name = f"softmax_{shape_str}_dim{dim}"

    @classmethod
    def get_test_configs(cls) -> List["SoftmaxTest"]:
        """Return all test configurations to run."""
        return [
            cls(shape=(2, 3, 4), dim=-1),  # softmax over last dimension
            cls(shape=(2, 3, 4), dim=1),  # softmax over middle dimension
            cls(shape=(4, 8), dim=-1),  # 2D tensor
            cls(shape=(2, 4, 8, 16), dim=-1),  # 4D tensor (common in transformers)
        ]

    def create_model(self) -> nn.Module:
        return SoftmaxModel(dim=self.dim)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        return (x,)

    def get_dynamic_shapes(self):
        return None


# Factory for CLI usage
def _create_from_args(args) -> SoftmaxTest:
    if args is None:
        return SoftmaxTest()
    shape = tuple(int(x) for x in args.shape.split(","))
    return SoftmaxTest(shape=shape, dim=args.dim)


def _add_args(parser):
    parser.add_argument(
        "--shape", type=str, default="2,3,4", help="Tensor shape (default: 2,3,4)"
    )
    parser.add_argument(
        "--dim", type=int, default=-1, help="Dimension for softmax (default: -1)"
    )


if __name__ == "__main__":
    run_op_test_main(_create_from_args, "Test softmax op on MLX delegate", _add_args)
