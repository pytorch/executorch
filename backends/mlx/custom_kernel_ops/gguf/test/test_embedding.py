#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for the GGUF Q6_K embedding lowering.

An ``nn.Embedding`` whose weight is an ``ExportableGGUFTensor`` exports to
``embedding(torchao::dequantize_gguf(weight, "q6_k", ...), indices)``. The MLX
``GGUF_QUANTIZED_EMBEDDING`` pattern matches that subgraph and lowers it to the
fused Q6_K gather Metal kernel. These tests compare the kernel against the eager
reference (``gguf``-package dequant + ``F.embedding``) on the same packed table.

Usage::

    python -m executorch.backends.mlx.custom_kernel_ops.gguf.test.test_embedding run
    python -m executorch.backends.mlx.custom_kernel_ops.gguf.test.test_embedding list
"""

from typing import List, Tuple

# Importing the patterns module registers GGUF_QUANTIZED_LINEAR / _EMBEDDING.
import executorch.backends.mlx.custom_kernel_ops.gguf.patterns  # noqa: F401
import torch
import torch.nn as nn
from executorch.backends.mlx.custom_kernel_ops.gguf.test.test_linear import (
    make_q6_k_blob,
)
from executorch.backends.mlx.test.test_utils import OpTestCase
from executorch.extension.llm.export.gguf import ExportableGGUFTensor


def _make_gguf_embedding_model(vocab: int, K: int, seed: int = 0) -> nn.Module:
    """An ``nn.Embedding`` whose weight is a Q6_K ``ExportableGGUFTensor``."""
    emb = nn.Embedding(vocab, K)
    blob = make_q6_k_blob(vocab, K, seed=seed)
    emb.weight = nn.Parameter(
        ExportableGGUFTensor.from_raw(blob, "q6_k", torch.bfloat16),
        requires_grad=False,
    )
    return emb


class GGUFEmbeddingTest(OpTestCase):
    name = "gguf_embedding"
    # Reference dequant runs in fp32 (gguf) then casts to bf16; the kernel
    # dequantizes per element to bf16, so allow bf16 tolerance.
    rtol = 2e-2
    atol = 2e-2

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
            # Real Gemma-4-31B embed width (K=5376, 21 Q6_K blocks/row). Vocab is
            # kept small so the packed weight fits CI-runner GPU buffer limits; the
            # gather + per-row dequant path is identical regardless of vocab.
            cls(vocab=2048, K=5376, idx_shape=(8,)),
        ]

    def get_edge_compile_config(self):
        from executorch.exir import EdgeCompileConfig

        # The dequantize_gguf custom op isn't a core ATen op; skip IR validity.
        return EdgeCompileConfig(_check_ir_validity=False)

    def create_model(self) -> nn.Module:
        return _make_gguf_embedding_model(self.vocab, self.K)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        torch.manual_seed(0)
        indices = torch.randint(0, self.vocab, self.idx_shape, dtype=torch.int64)
        return (indices,)


def _main() -> None:  # noqa: C901
    import argparse
    import sys

    from executorch.backends.mlx.test.test_utils import rebuild_op_test_runner

    parser = argparse.ArgumentParser(description="Test GGUF Q6_K embedding lowering")
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
