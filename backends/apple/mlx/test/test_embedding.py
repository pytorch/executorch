#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for nn.Embedding op using the MLX delegate.

Usage:
    # Run via run_all_tests (recommended):
    python -m executorch.backends.apple.mlx.test.run_all_tests embedding

    # Run directly with custom args:
    python -m executorch.backends.apple.mlx.test.test_embedding run --vocab-size 1000
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test, run_op_test_main


class EmbeddingModel(nn.Module):
    """Simple embedding layer for testing."""

    def __init__(
        self,
        num_embeddings: int = 1000,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


@register_test
class EmbeddingTest(OpTestCase):
    """Test case for nn.Embedding."""

    name = "embedding"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        num_embeddings: int = 1000,
        embedding_dim: int = 64,
        batch_size: int = 2,
        seq_len: int = 16,
    ):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.name = "embedding"

    @classmethod
    def get_test_configs(cls) -> List["EmbeddingTest"]:
        """Return all test configurations to run."""
        return [
            cls(),  # default config
            cls(num_embeddings=512, embedding_dim=128),  # different sizes
        ]

    def create_model(self) -> nn.Module:
        return EmbeddingModel(self.num_embeddings, self.embedding_dim)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # Random token indices
        x = torch.randint(0, self.num_embeddings, (self.batch_size, self.seq_len))
        return (x,)

    def get_dynamic_shapes(self):
        return None


# Factory for CLI usage
def _create_from_args(args) -> EmbeddingTest:
    if args is None:
        return EmbeddingTest()
    return EmbeddingTest(
        num_embeddings=args.vocab_size,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )


def _add_args(parser):
    parser.add_argument("--vocab-size", type=int, default=1000, help="Vocabulary size")
    parser.add_argument(
        "--embedding-dim", type=int, default=64, help="Embedding dimension"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length")


if __name__ == "__main__":
    run_op_test_main(_create_from_args, "Test nn.Embedding on MLX delegate", _add_args)
