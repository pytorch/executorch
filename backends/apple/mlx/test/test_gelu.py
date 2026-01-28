#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for GELU activation using the MLX delegate.

GELU (Gaussian Error Linear Unit) is commonly used in BERT, GPT, and many other models.

Usage:
    # Run via run_all_tests (recommended):
    python -m executorch.backends.apple.mlx.test.run_all_tests gelu

    # Run directly:
    python -m executorch.backends.apple.mlx.test.test_gelu run
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test, run_op_test_main


class GELUModel(nn.Module):
    """Simple model using GELU activation."""

    def __init__(self):
        super().__init__()
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu(x)


@register_test
class GELUTest(OpTestCase):
    """Test case for GELU activation."""

    name = "gelu"
    rtol = 1e-3
    atol = 1e-3

    def __init__(
        self,
        shape: Tuple[int, ...] = (2, 16, 64),
    ):
        self.shape = shape
        self.name = "gelu"

    @classmethod
    def get_test_configs(cls) -> List["GELUTest"]:
        """Return all test configurations to run."""
        return [
            cls(),  # default 3D shape
            cls(shape=(4, 32, 128)),  # larger shape
        ]

    def create_model(self) -> nn.Module:
        return GELUModel()

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        return (x,)

    def get_dynamic_shapes(self):
        return None


# Factory for CLI usage
def _create_from_args(args) -> GELUTest:
    if args is None:
        return GELUTest()
    shape = tuple(int(x) for x in args.shape.split(","))
    return GELUTest(shape=shape)


def _add_args(parser):
    parser.add_argument(
        "--shape", type=str, default="2,16,64", help="Tensor shape (default: 2,16,64)"
    )


if __name__ == "__main__":
    run_op_test_main(_create_from_args, "Test GELU on MLX delegate", _add_args)
