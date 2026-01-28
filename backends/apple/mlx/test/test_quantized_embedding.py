#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for TorchAO int4 quantized nn.Embedding using the MLX delegate.

This tests the QuantizedEmbeddingHandler pattern which fuses dequantize_affine
with embedding lookup for efficient quantized embedding tables.

Usage:
    # Run via run_all_tests (recommended):
    python -m executorch.backends.apple.mlx.test.run_all_tests quantized_embedding

    # Run directly with custom args:
    python -m executorch.backends.apple.mlx.test.test_quantized_embedding run --group-size 64
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test, run_op_test_main


class QuantizedEmbeddingModel(nn.Module):
    """Simple embedding layer that will be quantized."""

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
class QuantizedEmbeddingTest(OpTestCase):
    """Test case for TorchAO int4 quantized nn.Embedding."""

    name = "quantized_embedding"
    rtol = 0.1  # Higher tolerance for quantized ops
    atol = 0.1

    def __init__(
        self,
        num_embeddings: int = 1000,
        embedding_dim: int = 64,
        batch_size: int = 2,
        seq_len: int = 16,
        group_size: int = 32,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.group_size = group_size
        self.dtype = dtype

        # Build unique test name
        parts = ["quantized_embedding", f"g{group_size}"]
        self.name = "_".join(parts)

    @classmethod
    def get_test_configs(cls) -> List["QuantizedEmbeddingTest"]:
        """Return all test configurations to run."""
        return [
            cls(),  # default (group_size=32)
        ]

    def create_model(self) -> nn.Module:
        model = QuantizedEmbeddingModel(self.num_embeddings, self.embedding_dim)
        model = model.to(self.dtype)

        try:
            from torchao.quantization.granularity import PerGroup
            from torchao.quantization.quant_api import IntxWeightOnlyConfig, quantize_

            # Filter function to only quantize embedding layers
            def embedding_filter(module: nn.Module, fqn: str) -> bool:
                return isinstance(module, nn.Embedding)

            quantize_(
                model,
                IntxWeightOnlyConfig(
                    weight_dtype=torch.int4, granularity=PerGroup(self.group_size)
                ),
                embedding_filter,
            )
        except ImportError:
            raise RuntimeError("TorchAO not installed. Run: pip install torchao")

        return model

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # Random token indices
        x = torch.randint(0, self.num_embeddings, (self.batch_size, self.seq_len))
        return (x,)

    def get_dynamic_shapes(self):
        return None


# Factory for CLI usage
def _create_from_args(args) -> QuantizedEmbeddingTest:
    if args is None:
        return QuantizedEmbeddingTest()
    return QuantizedEmbeddingTest(
        num_embeddings=args.vocab_size,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        group_size=args.group_size,
    )


def _add_args(parser):
    parser.add_argument("--vocab-size", type=int, default=1000, help="Vocabulary size")
    parser.add_argument(
        "--embedding-dim", type=int, default=64, help="Embedding dimension"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length")
    parser.add_argument(
        "--group-size", type=int, default=32, help="Quantization group size"
    )


if __name__ == "__main__":
    run_op_test_main(
        _create_from_args, "Test quantized nn.Embedding on MLX delegate", _add_args
    )
