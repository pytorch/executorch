#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for nn.Conv1d op using the MLX delegate.

Conv1d is used in models like Whisper for audio processing.

Usage:
    # Run via run_all_tests (recommended):
    python -m executorch.backends.apple.mlx.test.run_all_tests conv1d

    # Run directly:
    python -m executorch.backends.apple.mlx.test.test_conv1d run
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test, run_op_test_main


class Conv1dModel(nn.Module):
    """Simple model using Conv1d."""

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 32,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@register_test
class Conv1dTest(OpTestCase):
    """Test case for nn.Conv1d."""

    name = "conv1d"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 32,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        batch_size: int = 2,
        seq_len: int = 64,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.batch_size = batch_size
        self.seq_len = seq_len

        # Build unique test name
        parts = ["conv1d"]
        if not bias:
            parts.append("no_bias")
        self.name = "_".join(parts)

    @classmethod
    def get_test_configs(cls) -> List["Conv1dTest"]:
        """Return all test configurations to run."""
        return [
            cls(),  # default with bias
            cls(bias=False),  # without bias
        ]

    def create_model(self) -> nn.Module:
        return Conv1dModel(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.bias,
        )

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # Conv1d expects (N, C_in, L)
        x = torch.randn(self.batch_size, self.in_channels, self.seq_len)
        return (x,)

    def get_dynamic_shapes(self):
        return None


# Factory for CLI usage
def _create_from_args(args) -> Conv1dTest:
    if args is None:
        return Conv1dTest()
    return Conv1dTest(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        kernel_size=args.kernel_size,
        stride=args.stride,
        padding=args.padding,
        bias=not args.no_bias,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )


def _add_args(parser):
    parser.add_argument("--in-channels", type=int, default=16, help="Input channels")
    parser.add_argument("--out-channels", type=int, default=32, help="Output channels")
    parser.add_argument("--kernel-size", type=int, default=3, help="Kernel size")
    parser.add_argument("--stride", type=int, default=1, help="Stride")
    parser.add_argument("--padding", type=int, default=1, help="Padding")
    parser.add_argument("--no-bias", action="store_true", help="Test without bias")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")


if __name__ == "__main__":
    run_op_test_main(_create_from_args, "Test nn.Conv1d on MLX delegate", _add_args)
