#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for ``mlx::int8_sym_int4_gs256_gemm`` (the M5 Max study's group-wise
int8 x int4 kernel): quantize + per-SIMD-group GEMM + split-K reduce or slice,
against the eager reference, over the default and pinned (Muse Glimmer ``o``)
tiers, split and unsplit tiles, padded M, activation dtypes, and dynamic M.

Usage::

    python -m executorch.backends.mlx.custom_kernel_ops.test.test_int8_sym_int4_gs256 run
    python -m executorch.backends.mlx.custom_kernel_ops.test.test_int8_sym_int4_gs256 run -v
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from executorch.backends.mlx.custom_kernel_ops.int8_intx_gemm import pack_int4
from executorch.backends.mlx.custom_kernel_ops.int8_sym_int4_gs256 import (
    effective_splits,
    GROUP_SIZE,
    Int8SymInt4Gs256Linear,
    parse_config,
    plans,
)
from executorch.backends.mlx.test.test_utils import OpTestCase

_TOLERANCES = {
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (2e-2, 2e-2),
}


def _tier_kernel_count(n: int, k: int, config: str) -> int:
    # Quantize + GEMM, plus the reduce when the tile splits K.
    splits = effective_splits(k // GROUP_SIZE, parse_config(config)[5])
    return 3 if splits > 1 else 2


def _kernel_count(n: int, k: int, m: int, test_m: Optional[int]) -> int:
    """Static M emits only its tier; dynamic M emits every tier's branch."""
    tiers = plans(n, k)
    if test_m is not None:
        return sum(_tier_kernel_count(n, k, c) for _, c in tiers)
    return _tier_kernel_count(n, k, next(c for max_m, c in tiers if m <= max_m))


class Int8SymInt4Gs256Test(OpTestCase):
    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype = torch.float16,
        test_m: Optional[int] = None,
    ):
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.test_m = test_m
        dtype_name = str(dtype).removeprefix("torch.")
        dyn = f"_dyn{test_m}" if test_m is not None else ""
        self.name = f"int8_sym_int4_gs256_m{m}_n{n}_k{k}_{dtype_name}{dyn}"
        self.rtol, self.atol = _TOLERANCES[dtype]
        self.expected_node_counts = {"MetalKernelNode": _kernel_count(n, k, m, test_m)}

    @classmethod
    def get_test_configs(cls) -> List["Int8SymInt4Gs256Test"]:
        shapes = [
            # Default tiers: split 16x64 tiles up to M = 16 (padded rows), then
            # unsplit 32x64 tiles (sliced back to M).
            {"m": 1, "n": 256, "k": 1024},
            {"m": 5, "n": 256, "k": 1024},
            {"m": 16, "n": 128, "k": 2048, "dtype": torch.bfloat16},
            {"m": 33, "n": 256, "k": 1024},
            {"m": 100, "n": 192, "k": 512, "dtype": torch.bfloat16},
            # Muse Glimmer `o` shape: every pinned tier (k16 / k4 splits,
            # 16x128 tiles, unsplit 32x128 / 32x64 tiles).
            {"m": 1, "n": 6656, "k": 4096},
            {"m": 2, "n": 6656, "k": 4096},
            {"m": 8, "n": 6656, "k": 4096},
            {"m": 12, "n": 6656, "k": 4096},
            {"m": 40, "n": 6656, "k": 4096},
            {"m": 100, "n": 6656, "k": 4096},
            # Dynamic M with every tier compiled in.
            {"m": 8, "n": 256, "k": 1024, "test_m": 3},
            {"m": 8, "n": 256, "k": 1024, "test_m": 50},
        ]
        return [cls(**s) for s in shapes]

    def create_model(self) -> nn.Module:
        groups = self.k // GROUP_SIZE
        wq = torch.randint(-8, 8, (self.n, self.k), dtype=torch.int8)
        scale = torch.rand(self.n, groups) * 0.02 + 0.005
        zero = torch.rand(self.n, groups) * 4 - 2
        return Int8SymInt4Gs256Linear(pack_int4(wq), scale, zero)

    def _make_inputs(self, m: int) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(m, self.k, dtype=self.dtype)
        # One all-zero row exercises the eps clamp on the activation scale.
        x[m // 2] = 0
        return (x,)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return self._make_inputs(self.m)

    def create_test_inputs(self) -> Tuple[torch.Tensor, ...]:
        return self._make_inputs(self.test_m if self.test_m is not None else self.m)

    def get_dynamic_shapes(self) -> Optional[Dict]:
        if self.test_m is None:
            return None
        return {"x": {0: torch.export.Dim("m", min=1, max=512)}}


if __name__ == "__main__":  # noqa: C901
    import argparse
    import sys

    from executorch.backends.mlx.test.test_utils import rebuild_op_test_runner

    parser = argparse.ArgumentParser(description="Test mlx::int8_sym_int4_gs256_gemm")
    parser.add_argument("action", choices=["generate", "compare", "run", "list"])
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    if args.rebuild and not rebuild_op_test_runner(verbose=args.verbose):
        sys.exit(1)

    configs = Int8SymInt4Gs256Test.get_test_configs()

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
