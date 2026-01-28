#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for nn.Linear op using the MLX delegate.

Usage:
    # Run via run_all_tests (recommended):
    python -m executorch.backends.apple.mlx.test.run_all_tests linear

    # Run directly with custom args:
    python -m executorch.backends.apple.mlx.test.test_linear run --no-bias
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test, run_op_test_main


class LinearModel(nn.Module):
    """Simple linear layer for testing."""

    def __init__(
        self, in_features: int = 64, out_features: int = 128, bias: bool = True
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@register_test
class LinearTest(OpTestCase):
    """Test case for nn.Linear."""

    name = "linear"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        in_features: int = 64,
        out_features: int = 128,
        batch_size: int = 2,
        seq_len: int = 16,
        bias: bool = True,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.bias = bias

        # Build unique test name
        if not bias:
            self.name = "linear_no_bias"
        else:
            self.name = "linear"

    @classmethod
    def get_test_configs(cls) -> List["LinearTest"]:
        """Return all test configurations to run."""
        return [
            cls(),  # with bias
            cls(bias=False),  # without bias
        ]

    def create_model(self) -> nn.Module:
        return LinearModel(self.in_features, self.out_features, bias=self.bias)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.batch_size, self.seq_len, self.in_features)
        return (x,)

    def get_dynamic_shapes(self):
        return None


# Factory for CLI usage
def _create_from_args(args) -> LinearTest:
    if args is None:
        return LinearTest()
    return LinearTest(
        in_features=args.in_features,
        out_features=args.out_features,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        bias=not args.no_bias,
    )


def _add_args(parser):
    parser.add_argument("--in-features", type=int, default=64, help="Input features")
    parser.add_argument("--out-features", type=int, default=128, help="Output features")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length")
    parser.add_argument("--no-bias", action="store_true", help="Test without bias")


if __name__ == "__main__":
    run_op_test_main(_create_from_args, "Test nn.Linear on MLX delegate", _add_args)
