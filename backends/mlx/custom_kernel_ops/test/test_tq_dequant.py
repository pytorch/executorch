#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for ``mlx::tq_dequant``.

Verifies the fused unpack + gather + multiply Metal kernel matches
the eager reference at head_dim values used by TurboQuant
(D ∈ {128, 256, 512}). Output is byte-exact — no fp32 promotion in
either path.

Usage::

    python -m executorch.backends.mlx.custom_kernel_ops.test.test_tq_dequant run
    python -m executorch.backends.mlx.custom_kernel_ops.test.test_tq_dequant run -v
    python -m executorch.backends.mlx.custom_kernel_ops.test.test_tq_dequant run --rebuild
"""

from typing import List, Tuple

import executorch.backends.mlx.custom_kernel_ops.tq_dequant  # noqa: F401

import torch
import torch.nn as nn

from executorch.backends.mlx.test.test_utils import OpTestCase


class TQDequantModel(nn.Module):
    """``packed, norms, centroids → unrotated``."""

    def forward(
        self,
        packed: torch.Tensor,
        norms: torch.Tensor,
        centroids: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.mlx.tq_dequant(packed, norms, centroids)


class TQDequantTest(OpTestCase):
    """Byte-exact comparison vs eager unpack + gather + multiply."""

    name = "tq_dequant"
    rtol = 0.0
    atol = 0.0

    def __init__(
        self,
        batch_size: int = 1,
        n_heads: int = 8,
        seq_len: int = 4,
        head_dim: int = 128,
    ):
        self.batch_size = batch_size
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.half_dim = head_dim // 2
        self.name = f"tq_dequant_b{batch_size}_h{n_heads}_t{seq_len}_d{head_dim}"

    @classmethod
    def get_test_configs(cls) -> List["TQDequantTest"]:
        return [
            # head_dim=128 (Qwen3.5 MoE / Gemma 4 sliding)
            cls(seq_len=1, head_dim=128),
            cls(seq_len=8, head_dim=128),
            cls(seq_len=64, head_dim=128),
            cls(n_heads=1, seq_len=1, head_dim=128),
            # head_dim=256 (Gemma 4 sliding-attention)
            cls(seq_len=4, head_dim=256),
            cls(seq_len=16, head_dim=256),
            # head_dim=512 (Gemma 4 31B full-attention)
            cls(n_heads=4, seq_len=4, head_dim=512),
            cls(n_heads=4, seq_len=64, head_dim=512),
        ]

    def create_model(self) -> nn.Module:
        return TQDequantModel()

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # Random packed bytes exercise every codebook entry.
        packed = torch.randint(
            0,
            256,
            (self.batch_size, self.n_heads, self.seq_len, self.half_dim),
            dtype=torch.uint8,
        )
        norms = (
            torch.randn(
                self.batch_size,
                self.n_heads,
                self.seq_len,
                1,
                dtype=torch.bfloat16,
            ).abs()
            + 0.1
        )
        # Deterministic codebook covering [-1, 1].
        centroids = torch.linspace(-1.0, 1.0, 16, dtype=torch.bfloat16)
        return (packed, norms, centroids)


if __name__ == "__main__":  # noqa: C901
    import argparse
    import sys

    from executorch.backends.mlx.test.test_utils import rebuild_op_test_runner

    parser = argparse.ArgumentParser(description="Test mlx::tq_dequant op")
    parser.add_argument("action", choices=["generate", "compare", "run", "list"])
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    if args.rebuild and not rebuild_op_test_runner(verbose=args.verbose):
        sys.exit(1)

    configs = TQDequantTest.get_test_configs()

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
            if ok:
                passed += 1
            else:
                failed += 1
                failed_names.append(test.name)
        elif args.action == "run":
            ok = test.run_test(verbose=args.verbose)
            if ok:
                passed += 1
            else:
                failed += 1
                failed_names.append(test.name)

    if args.action in ("run", "compare"):
        print(f"\nPassed: {passed}, Failed: {failed}")
        if failed_names:
            print(f"Failed: {', '.join(failed_names)}")
        sys.exit(0 if failed == 0 else 1)
