#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for where op using the MLX delegate.

Usage:
    # Run via run_all_tests (recommended):
    python -m executorch.backends.apple.mlx.test.run_all_tests where

    # Run directly:
    python -m executorch.backends.apple.mlx.test.test_where run
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test, run_op_test_main


class WhereModel(nn.Module):
    """Model that conditionally selects from x or y based on condition."""

    def forward(
        self, condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return torch.where(condition, x, y)


@register_test
class WhereTest(OpTestCase):
    """Test case for where op."""

    name = "where"
    rtol = 1e-5
    atol = 1e-5

    def __init__(self, shape: Tuple[int, ...] = (2, 3, 4)):
        self.shape = shape
        shape_str = "x".join(str(s) for s in shape)
        self.name = f"where_{shape_str}"

    @classmethod
    def get_test_configs(cls) -> List["WhereTest"]:
        """Return all test configurations to run."""
        return [
            # 3D tensor (typical for sequences)
            cls(shape=(2, 3, 4)),
            # 2D tensor (batch x features)
            cls(shape=(4, 8)),
            # 4D tensor (transformers attention scores)
            cls(shape=(2, 8, 16, 16)),
            # Large tensor for attention masking
            cls(shape=(1, 1, 128, 128)),
        ]

    def create_model(self) -> nn.Module:
        return WhereModel()

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # Create condition tensor (boolean mask)
        condition = torch.rand(self.shape) > 0.5
        # Create input tensors with different values
        x = torch.randn(self.shape)
        y = torch.randn(self.shape)
        return (condition, x, y)

    def get_dynamic_shapes(self):
        return None


# Factory for CLI usage
def _create_from_args(args) -> WhereTest:
    if args is None:
        return WhereTest()
    shape = tuple(int(x) for x in args.shape.split(","))
    return WhereTest(shape=shape)


def _add_args(parser):
    parser.add_argument(
        "--shape",
        type=str,
        default="2,3,4",
        help="Tensor shape (default: 2,3,4)",
    )


if __name__ == "__main__":
    run_op_test_main(_create_from_args, "Test where op on MLX delegate", _add_args)
