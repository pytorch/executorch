#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for torch.arange() op using the MLX delegate.

Arange creates a 1D tensor with evenly spaced values.

Usage:
    # Run via run_all_tests (recommended):
    python -m executorch.backends.apple.mlx.test.run_all_tests arange

    # Run directly:
    python -m executorch.backends.apple.mlx.test.test_arange run
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test, run_op_test_main


class ArangeModel(nn.Module):
    """Model that creates a tensor using arange and multiplies with input."""

    def __init__(self, stop: int):
        super().__init__()
        self.stop = stop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create arange and multiply with input to ensure arange goes through graph
        indices = torch.arange(self.stop, dtype=x.dtype, device=x.device)
        return x * indices


@register_test
class ArangeTest(OpTestCase):
    """Test case for torch.arange()."""

    name = "arange"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        stop: int = 10,
    ):
        self.stop = stop
        self.name = f"arange_{stop}"

    @classmethod
    def get_test_configs(cls) -> List["ArangeTest"]:
        """Return all test configurations to run."""
        return [
            cls(stop=10),
            cls(stop=32),
            cls(stop=100),
        ]

    def create_model(self) -> nn.Module:
        return ArangeModel(self.stop)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.stop)
        return (x,)

    def get_dynamic_shapes(self):
        return None


# Factory for CLI usage
def _create_from_args(args) -> ArangeTest:
    if args is None:
        return ArangeTest()
    return ArangeTest(
        stop=args.stop,
    )


def _add_args(parser):
    parser.add_argument("--stop", type=int, default=10, help="Stop value for arange")


if __name__ == "__main__":
    run_op_test_main(
        _create_from_args, "Test torch.arange() on MLX delegate", _add_args
    )
