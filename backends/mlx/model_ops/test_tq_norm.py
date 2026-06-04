#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for ``mlx::tq_norm``.

Verifies the fused L2-norm Metal kernel matches eager ``vector_norm``
at head_dim values used by TurboQuant (D ∈ {128, 256, 512}).

Usage::

    python -m executorch.backends.mlx.model_ops.test_tq_norm run
    python -m executorch.backends.mlx.model_ops.test_tq_norm run -v
    python -m executorch.backends.mlx.model_ops.test_tq_norm run --rebuild
"""

from typing import List, Tuple

import executorch.backends.mlx.model_ops.tq_norm  # noqa: F401

import torch
import torch.nn as nn

from executorch.backends.mlx.test.test_utils import OpTestCase


class TQNormModel(nn.Module):
    """``x → ||x||₂`` over the last dim."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.mlx.tq_norm(x)


class TQNormTest(OpTestCase):
    """Compare ``mlx::tq_norm`` to eager ``vector_norm`` within bf16 ULPs."""

    name = "tq_norm"
    rtol = 1e-2
    atol = 1e-2

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
        self.name = f"tq_norm_b{batch_size}_h{n_heads}_t{seq_len}_d{head_dim}"

    @classmethod
    def get_test_configs(cls) -> List["TQNormTest"]:
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
        return TQNormModel().to(torch.bfloat16)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # Activation-scale bf16 inputs.
        x = torch.randn(
            self.batch_size,
            self.n_heads,
            self.seq_len,
            self.head_dim,
            dtype=torch.bfloat16,
        ) * (1.0 / (self.head_dim**0.5))
        return (x,)


if __name__ == "__main__":  # noqa: C901
    import argparse
    import sys

    from executorch.backends.mlx.test.test_utils import rebuild_op_test_runner

    parser = argparse.ArgumentParser(description="Test mlx::tq_norm op")
    parser.add_argument(
        "action",
        choices=["generate", "compare", "run", "list"],
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    if args.rebuild and not rebuild_op_test_runner(verbose=args.verbose):
        sys.exit(1)

    configs = TQNormTest.get_test_configs()

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
