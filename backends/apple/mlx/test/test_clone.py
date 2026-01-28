#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for tensor.clone() op using the MLX delegate.

Clone creates a copy of the tensor with the same data.

Usage:
    # Run via run_all_tests (recommended):
    python -m executorch.backends.apple.mlx.test.run_all_tests clone

    # Run directly:
    python -m executorch.backends.apple.mlx.test.test_clone run
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test, run_op_test_main


class CloneModel(nn.Module):
    """Model that clones a tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clone()


@register_test
class CloneTest(OpTestCase):
    """Test case for tensor.clone()."""

    name = "clone"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        shape: Tuple[int, ...] = (2, 3, 4),
    ):
        self.shape = shape
        shape_str = "x".join(str(d) for d in shape)
        self.name = f"clone_{shape_str}"

    @classmethod
    def get_test_configs(cls) -> List["CloneTest"]:
        """Return all test configurations to run."""
        return [
            cls(shape=(2, 3, 4)),  # 3D
            cls(shape=(8, 8)),  # 2D
            cls(shape=(16,)),  # 1D
        ]

    def create_model(self) -> nn.Module:
        return CloneModel()

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        return (x,)

    def get_dynamic_shapes(self):
        return None


# Factory for CLI usage
def _create_from_args(args) -> CloneTest:
    if args is None:
        return CloneTest()
    return CloneTest(
        shape=tuple(args.shape),
    )


def _add_args(parser):
    parser.add_argument(
        "--shape", type=int, nargs="+", default=[2, 3, 4], help="Input shape"
    )


if __name__ == "__main__":
    run_op_test_main(
        _create_from_args, "Test tensor.clone() on MLX delegate", _add_args
    )
