#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for nn.LayerNorm op using the MLX delegate.

LayerNorm is commonly used in BERT, GPT, and other transformer models.

Usage:
    # Run via run_all_tests (recommended):
    python -m executorch.backends.apple.mlx.test.run_all_tests layer_norm

    # Run directly:
    python -m executorch.backends.apple.mlx.test.test_layer_norm run
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test, run_op_test_main


class LayerNormModel(nn.Module):
    """Simple model using LayerNorm."""

    def __init__(self, normalized_shape: int = 64, eps: float = 1e-5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(x)


@register_test
class LayerNormTest(OpTestCase):
    """Test case for nn.LayerNorm."""

    name = "layer_norm"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        normalized_shape: int = 64,
        batch_size: int = 2,
        seq_len: int = 16,
        eps: float = 1e-5,
    ):
        self.normalized_shape = normalized_shape
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.eps = eps
        self.name = "layer_norm"

    @classmethod
    def get_test_configs(cls) -> List["LayerNormTest"]:
        """Return all test configurations to run."""
        return [
            cls(),  # default config
            cls(normalized_shape=128, eps=1e-6),  # different hidden dim and eps
        ]

    def create_model(self) -> nn.Module:
        return LayerNormModel(self.normalized_shape, self.eps)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.batch_size, self.seq_len, self.normalized_shape)
        return (x,)

    def get_dynamic_shapes(self):
        return None


# Factory for CLI usage
def _create_from_args(args) -> LayerNormTest:
    if args is None:
        return LayerNormTest()
    return LayerNormTest(
        normalized_shape=args.normalized_shape,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        eps=args.eps,
    )


def _add_args(parser):
    parser.add_argument(
        "--normalized-shape", type=int, default=64, help="Normalized shape (last dim)"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length")
    parser.add_argument("--eps", type=float, default=1e-5, help="Epsilon for LayerNorm")


if __name__ == "__main__":
    run_op_test_main(_create_from_args, "Test nn.LayerNorm on MLX delegate", _add_args)
