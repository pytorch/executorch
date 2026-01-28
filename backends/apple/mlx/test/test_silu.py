#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for SiLU (Swish) activation using the MLX delegate.

SiLU (Sigmoid Linear Unit) is defined as: silu(x) = x * sigmoid(x)
It's commonly used in modern LLMs like LLaMA, Mistral, etc.

Usage:
    # Run via run_all_tests (recommended):
    python -m executorch.backends.apple.mlx.test.run_all_tests silu

    # Run directly:
    python -m executorch.backends.apple.mlx.test.test_silu run
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test, run_op_test_main


class SiLUModel(nn.Module):
    """Simple model using SiLU activation."""

    def __init__(self):
        super().__init__()
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.silu(x)


@register_test
class SiLUTest(OpTestCase):
    """Test case for SiLU activation."""

    name = "silu"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        shape: Tuple[int, ...] = (2, 16, 64),
    ):
        self.shape = shape
        self.name = "silu"

    @classmethod
    def get_test_configs(cls) -> List["SiLUTest"]:
        """Return all test configurations to run."""
        return [
            cls(),  # default 3D shape
            cls(shape=(4, 32, 128)),  # larger shape
        ]

    def create_model(self) -> nn.Module:
        return SiLUModel()

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        return (x,)

    def get_dynamic_shapes(self):
        return None


# Factory for CLI usage
def _create_from_args(args) -> SiLUTest:
    if args is None:
        return SiLUTest()
    shape = tuple(int(x) for x in args.shape.split(","))
    return SiLUTest(shape=shape)


def _add_args(parser):
    parser.add_argument(
        "--shape", type=str, default="2,16,64", help="Tensor shape (default: 2,16,64)"
    )


if __name__ == "__main__":
    run_op_test_main(_create_from_args, "Test SiLU on MLX delegate", _add_args)
