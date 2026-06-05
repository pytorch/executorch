#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for ``mlx::gguf_embedding`` (GGUF Q6_K embedding gather).

Compares the fused gather Metal kernel against the eager reference on the same
packed Q6_K table. The kernel and reference run identical per-element float
dequant, so the bf16 outputs match exactly.

Usage::

    python -m executorch.backends.mlx.custom_kernel_ops.gguf.test.test_embedding run
    python -m executorch.backends.mlx.custom_kernel_ops.gguf.test.test_embedding run --rebuild
"""

from typing import List, Tuple

import executorch.backends.mlx.custom_kernel_ops.gguf.embedding  # noqa: F401

import torch
import torch.nn as nn

from executorch.backends.mlx.custom_kernel_ops.gguf.test.test_linear import (
    make_q6_k_blob,
)
from executorch.backends.mlx.test.test_utils import OpTestCase


class GGUFEmbeddingModel(nn.Module):
    def forward(self, weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return torch.ops.mlx.gguf_embedding(weight, indices, "q6k")


class GGUFEmbeddingTest(OpTestCase):
    name = "gguf_embedding"
    rtol = 0.0
    atol = 0.0

    def __init__(
        self,
        vocab: int = 512,
        K: int = 256,
        idx_shape: Tuple[int, ...] = (8,),
    ):
        self.vocab = vocab
        self.K = K
        self.idx_shape = idx_shape
        shp = "x".join(str(d) for d in idx_shape)
        self.name = f"gguf_embedding_v{vocab}_k{K}_idx{shp}"

    @classmethod
    def get_test_configs(cls) -> List["GGUFEmbeddingTest"]:
        return [
            cls(vocab=512, K=256, idx_shape=(1,)),
            cls(vocab=512, K=256, idx_shape=(8,)),
            cls(vocab=512, K=256, idx_shape=(64,)),
            cls(vocab=512, K=512, idx_shape=(8,)),
            cls(vocab=512, K=1024, idx_shape=(4,)),
            cls(vocab=300, K=256, idx_shape=(16,)),  # vocab not tile-aligned
            cls(vocab=512, K=256, idx_shape=(2, 3)),  # multi-dim indices
            cls(vocab=262144, K=5376, idx_shape=(8,)),  # real Gemma-4-31B embed
        ]

    def create_model(self) -> nn.Module:
        return GGUFEmbeddingModel()

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        torch.manual_seed(0)
        weight = make_q6_k_blob(self.vocab, self.K)
        indices = torch.randint(0, self.vocab, self.idx_shape, dtype=torch.int32)
        return (weight, indices)


def _main() -> None:  # noqa: C901
    import argparse
    import sys

    from executorch.backends.mlx.test.test_utils import rebuild_op_test_runner

    parser = argparse.ArgumentParser(description="Test mlx::gguf_embedding op")
    parser.add_argument("action", choices=["generate", "compare", "run", "list"])
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    if args.rebuild and not rebuild_op_test_runner(verbose=args.verbose):
        sys.exit(1)

    configs = GGUFEmbeddingTest.get_test_configs()

    if args.action == "list":
        for cfg in configs:
            print(f"  {cfg.name}")
        sys.exit(0)

    if args.config:
        configs = [c for c in configs if c.name == args.config]
        if not configs:
            print(f"No config matching '{args.config}'")
            sys.exit(1)

    passed = 0
    failed = 0
    failed_names: List[str] = []

    for test in configs:
        if args.action == "generate":
            pte_path, _, _ = test.generate_test_files(verbose=args.verbose)
            print(f"Generated: {pte_path}")
        elif args.action == "compare":
            actual_path = test.get_test_dir() / "actual_output.bin"
            ok, msg = test.compare_with_actual(actual_path)
            print(f"{'✓' if ok else '✗'} {test.name}: {msg}")
            passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)
            if not ok:
                failed_names.append(test.name)
        elif args.action == "run":
            ok = test.run_test(verbose=args.verbose)
            passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)
            if not ok:
                failed_names.append(test.name)

    if args.action in ("run", "compare"):
        print(f"\nPassed: {passed}, Failed: {failed}")
        if failed_names:
            print(f"Failed: {', '.join(failed_names)}")
        sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    _main()
