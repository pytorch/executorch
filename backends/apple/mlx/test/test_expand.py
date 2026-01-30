#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for expand (expand_copy) op using the MLX delegate.

Usage:
    # Run via run_all_tests (recommended):
    python -m executorch.backends.apple.mlx.test.run_all_tests expand

    # Run directly:
    python -m executorch.backends.apple.mlx.test.test_expand run
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test, run_op_test_main


class ExpandModel(nn.Module):
    """Model that expands a tensor to a larger shape."""

    def __init__(self, target_shape: Tuple[int, ...]):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.expand(self.target_shape)


@register_test
class ExpandTest(OpTestCase):
    """Test case for expand (expand_copy) op."""

    name = "expand"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        input_shape: Tuple[int, ...] = (1, 3, 1),
        target_shape: Tuple[int, ...] = (2, 3, 4),
    ):
        self.input_shape = input_shape
        self.target_shape = target_shape

        # Build unique test name
        input_str = "x".join(str(s) for s in input_shape)
        target_str = "x".join(str(s) for s in target_shape)
        self.name = f"expand_{input_str}_to_{target_str}"

    @classmethod
    def get_test_configs(cls) -> List["ExpandTest"]:
        """Return all test configurations to run."""
        return [
            # Expand last dimension: (2, 3, 1) -> (2, 3, 4)
            cls(input_shape=(2, 3, 1), target_shape=(2, 3, 4)),
            # Expand first dimension: (1, 3, 4) -> (2, 3, 4)
            cls(input_shape=(1, 3, 4), target_shape=(2, 3, 4)),
            # Expand multiple dimensions: (1, 1, 4) -> (2, 3, 4)
            cls(input_shape=(1, 1, 4), target_shape=(2, 3, 4)),
            # Expand all dimensions: (1, 1, 1) -> (2, 3, 4)
            cls(input_shape=(1, 1, 1), target_shape=(2, 3, 4)),
            # 2D case: (1, 8) -> (4, 8)
            cls(input_shape=(1, 8), target_shape=(4, 8)),
            # 4D case (transformers): (1, 1, 1, 64) -> (2, 8, 16, 64)
            cls(input_shape=(1, 1, 1, 64), target_shape=(2, 8, 16, 64)),
        ]

    def create_model(self) -> nn.Module:
        return ExpandModel(self.target_shape)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.input_shape)
        return (x,)

    def get_dynamic_shapes(self):
        return None


# Factory for CLI usage
def _create_from_args(args) -> ExpandTest:
    if args is None:
        return ExpandTest()
    input_shape = tuple(int(x) for x in args.input_shape.split(","))
    target_shape = tuple(int(x) for x in args.target_shape.split(","))
    return ExpandTest(input_shape=input_shape, target_shape=target_shape)


def _add_args(parser):
    parser.add_argument(
        "--input-shape",
        type=str,
        default="1,3,1",
        help="Input tensor shape (default: 1,3,1)",
    )
    parser.add_argument(
        "--target-shape",
        type=str,
        default="2,3,4",
        help="Target expand shape (default: 2,3,4)",
    )


if __name__ == "__main__":
    run_op_test_main(_create_from_args, "Test expand op on MLX delegate", _add_args)
