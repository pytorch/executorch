#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for tensor.repeat() op using the MLX delegate.

Usage:
    # Run via run_all_tests (recommended):
    python -m executorch.backends.apple.mlx.test.run_all_tests repeat

    # Run directly:
    python -m executorch.backends.apple.mlx.test.test_repeat run
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test, run_op_test_main


class RepeatModel(nn.Module):
    """Model that repeats a tensor along specified dimensions."""

    def __init__(self, repeats: Tuple[int, ...]):
        super().__init__()
        self.repeats = repeats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat(*self.repeats)


@register_test
class RepeatTest(OpTestCase):
    """Test case for tensor.repeat()."""

    name = "repeat"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        input_shape: Tuple[int, ...] = (2, 3, 4),
        repeats: Tuple[int, ...] = (2, 1, 3),
    ):
        self.input_shape = input_shape
        self.repeats = repeats

        # Build unique test name
        repeat_str = "x".join(str(r) for r in repeats)
        self.name = f"repeat_{repeat_str}"

    @classmethod
    def get_test_configs(cls) -> List["RepeatTest"]:
        """Return all test configurations to run."""
        return [
            cls(input_shape=(2, 3), repeats=(2, 3)),  # 2D repeat
            cls(input_shape=(2, 3, 4), repeats=(1, 2, 1)),  # repeat middle dim
            cls(input_shape=(4, 4), repeats=(3, 3)),  # square repeat
        ]

    def create_model(self) -> nn.Module:
        return RepeatModel(self.repeats)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.input_shape)
        return (x,)

    def get_dynamic_shapes(self):
        return None


# Factory for CLI usage
def _create_from_args(args) -> RepeatTest:
    if args is None:
        return RepeatTest()
    return RepeatTest(
        input_shape=tuple(args.input_shape),
        repeats=tuple(args.repeats),
    )


def _add_args(parser):
    parser.add_argument(
        "--input-shape", type=int, nargs="+", default=[2, 3, 4], help="Input shape"
    )
    parser.add_argument(
        "--repeats", type=int, nargs="+", default=[2, 1, 3], help="Repeat factors"
    )


if __name__ == "__main__":
    run_op_test_main(
        _create_from_args, "Test tensor.repeat() on MLX delegate", _add_args
    )
