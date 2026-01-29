#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for add op using the MLX delegate.

Usage:
    # Run via run_all_tests (recommended):
    python -m executorch.backends.apple.mlx.test.run_all_tests add

    # Run directly with custom args:
    python -m executorch.backends.apple.mlx.test.test_add run --scalar 2.5
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test, run_op_test_main


class AddTensorModel(nn.Module):
    """Add two tensors."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class AddScalarModel(nn.Module):
    """Add tensor and scalar."""

    def __init__(self, scalar: float = 1.0):
        super().__init__()
        self.scalar = scalar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scalar


@register_test
class AddTest(OpTestCase):
    """Test case for add op."""

    name = "add"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        shape: Tuple[int, ...] = (2, 16, 64),
        scalar: Optional[float] = None,
    ):
        self.shape = shape
        self.scalar = scalar

        # Build unique test name
        if scalar is not None:
            self.name = "add_scalar"
        else:
            self.name = "add"

    @classmethod
    def get_test_configs(cls) -> List["AddTest"]:
        """Return all test configurations to run."""
        return [
            cls(),  # tensor + tensor
            cls(scalar=2.5),  # tensor + scalar
        ]

    def create_model(self) -> nn.Module:
        if self.scalar is not None:
            return AddScalarModel(self.scalar)
        else:
            return AddTensorModel()

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        if self.scalar is not None:
            return (x,)
        else:
            y = torch.randn(self.shape)
            return (x, y)

    def get_dynamic_shapes(self):
        return None


# Factory for CLI usage
def _create_from_args(args) -> AddTest:
    if args is None:
        return AddTest()
    shape = tuple(int(x) for x in args.shape.split(","))
    return AddTest(shape=shape, scalar=args.scalar)


def _add_args(parser):
    parser.add_argument(
        "--shape", type=str, default="2,16,64", help="Tensor shape (default: 2,16,64)"
    )
    parser.add_argument(
        "--scalar", type=float, default=None, help="Test tensor + scalar"
    )


if __name__ == "__main__":
    run_op_test_main(_create_from_args, "Test add op on MLX delegate", _add_args)
