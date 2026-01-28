#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for tensor indexing with tensor indices using the MLX delegate.

This tests aten.index.Tensor which uses take_along_axis in MLX.

Usage:
    # Run via run_all_tests (recommended):
    python -m executorch.backends.apple.mlx.test.run_all_tests index

    # Run directly:
    python -m executorch.backends.apple.mlx.test.test_index run
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test, run_op_test_main


class IndexModel(nn.Module):
    """Model that indexes a tensor using another tensor."""

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return x[indices]


@register_test
class IndexTest(OpTestCase):
    """Test case for tensor indexing.

    Note: The MLX _index_handler uses take_along_axis which requires
    indices to have the same number of dimensions as the source tensor.
    This test uses 1D tensors for both source and indices.
    """

    name = "index"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        table_size: int = 100,
        num_indices: int = 10,
    ):
        self.table_size = table_size
        self.num_indices = num_indices
        self.name = f"index_{table_size}_idx{num_indices}"

    @classmethod
    def get_test_configs(cls) -> List["IndexTest"]:
        """Return all test configurations to run."""
        return [
            cls(table_size=100, num_indices=10),
            cls(table_size=50, num_indices=5),
        ]

    def create_model(self) -> nn.Module:
        return IndexModel()

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # 1D tensor for both x and indices (take_along_axis requires same ndim)
        x = torch.randn(self.table_size)
        indices = torch.randint(0, self.table_size, (self.num_indices,))
        return (x, indices)

    def get_dynamic_shapes(self):
        return None


# Factory for CLI usage
def _create_from_args(args) -> IndexTest:
    if args is None:
        return IndexTest()
    return IndexTest(
        table_size=args.table_size,
        num_indices=args.num_indices,
    )


def _add_args(parser):
    parser.add_argument(
        "--table-size", type=int, default=100, help="Size of table to index into"
    )
    parser.add_argument(
        "--num-indices", type=int, default=10, help="Number of indices to lookup"
    )


if __name__ == "__main__":
    run_op_test_main(
        _create_from_args, "Test tensor indexing on MLX delegate", _add_args
    )
