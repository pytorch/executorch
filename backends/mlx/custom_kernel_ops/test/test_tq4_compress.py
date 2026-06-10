#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for ``mlx::tq4_compress``.

Verifies the fused Metal kernel produces byte-exact output vs the
eager Python implementation across head_dim values used by TurboQuant.

Usage::

    python -m executorch.backends.mlx.custom_kernel_ops.test.test_tq4_compress run
    python -m executorch.backends.mlx.custom_kernel_ops.test.test_tq4_compress run -v
    python -m executorch.backends.mlx.custom_kernel_ops.test.test_tq4_compress run --rebuild
"""

from typing import List, Tuple

import executorch.backends.mlx.custom_kernel_ops.tq4_compress  # noqa: F401

import torch
import torch.nn as nn

from executorch.backends.mlx.test.test_utils import OpTestCase


class TQ4CompressModel(nn.Module):
    """``values → packed`` via ``mlx::tq4_compress``.

    Boundaries are stored as a buffer so the model is exportable
    without feeding them as a graph input.
    """

    def __init__(self, head_dim: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        # 15 sorted thresholds (4-bit codebook).
        self.register_buffer(
            "boundaries",
            torch.linspace(-0.2, 0.2, 15, dtype=dtype),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.ops.mlx.tq4_compress(values, self.boundaries)


class TQ4CompressTest(OpTestCase):
    """Byte-exact comparison vs eager bucketize + nibble-pack."""

    name = "tq4_compress"
    rtol = 0.0
    atol = 0.0

    def __init__(
        self,
        batch_size: int = 1,
        n_heads: int = 8,
        seq_len: int = 4,
        head_dim: int = 128,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.batch_size = batch_size
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.dtype = dtype

        parts = [
            "tq4_compress",
            f"b{batch_size}",
            f"h{n_heads}",
            f"t{seq_len}",
            f"d{head_dim}",
        ]
        if dtype != torch.bfloat16:
            parts.append(str(dtype).split(".")[-1])
        self.name = "_".join(parts)

    @classmethod
    def get_test_configs(cls) -> List["TQ4CompressTest"]:
        return [
            # head_dim=128 (Qwen3.5 MoE / Gemma 4 sliding)
            cls(seq_len=1, head_dim=128),
            cls(seq_len=8, head_dim=128),
            cls(seq_len=64, head_dim=128),
            cls(n_heads=1, seq_len=1, head_dim=128),
            # head_dim=256 (Gemma 4 sliding-attention)
            cls(head_dim=256),
            cls(seq_len=16, head_dim=256),
            # head_dim=512 (Gemma 4 31B full-attention)
            cls(n_heads=4, seq_len=4, head_dim=512),
            cls(n_heads=4, seq_len=64, head_dim=512),
            # Smaller D for sanity
            cls(head_dim=64, n_heads=2, seq_len=4),
        ]

    def create_model(self) -> nn.Module:
        return TQ4CompressModel(head_dim=self.head_dim, dtype=self.dtype).to(self.dtype)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # Activation-scale values; the kernel is byte-exact regardless
        # of magnitude as long as values fall within the bucketize
        # comparison range.
        values = torch.randn(
            self.batch_size,
            self.n_heads,
            self.seq_len,
            self.head_dim,
            dtype=self.dtype,
        ) * (1.0 / (self.head_dim**0.5))
        return (values,)


if __name__ == "__main__":  # noqa: C901
    import argparse
    import sys

    from executorch.backends.mlx.test.test_utils import rebuild_op_test_runner

    parser = argparse.ArgumentParser(description="Test mlx::tq4_compress op")
    parser.add_argument(
        "action",
        choices=["generate", "compare", "run", "list"],
        help="Action: generate (export), compare (check outputs), run (full), list (show configs)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--rebuild", action="store_true", help="Rebuild C++ runner first"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Run specific config by name"
    )
    args = parser.parse_args()

    if args.rebuild and not rebuild_op_test_runner(verbose=args.verbose):
        sys.exit(1)

    configs = TQ4CompressTest.get_test_configs()

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
