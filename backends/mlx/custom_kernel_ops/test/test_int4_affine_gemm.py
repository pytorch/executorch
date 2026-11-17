#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for ``mlx::int8_int4_affine_gemm`` and ``mlx::fp_int4_affine_gemm``.

Compares the prologue + runtime-selected matvec / grouped full-K / split-K
GEMM against the eager reference for int8 and fp activations, group sizes 64,
128, 256 and per-channel, over M/N tile tails, activation dtypes, and dynamic
M.

Usage::

    python -m executorch.backends.mlx.custom_kernel_ops.test.test_int4_affine_gemm run
    python -m executorch.backends.mlx.custom_kernel_ops.test.test_int4_affine_gemm run -v
    python -m executorch.backends.mlx.custom_kernel_ops.test.test_int4_affine_gemm run --rebuild
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from executorch.backends.mlx.custom_kernel_ops.int4_affine_gemm import (
    ACTS,
    affine_plans,
    AffinePlan,
    GROUP_SIZES,
    Int4AffineLinear,
)
from executorch.backends.mlx.custom_kernel_ops.int8_intx_gemm import pack_int4
from executorch.backends.mlx.test.test_utils import OpTestCase

_TOLERANCES = {
    torch.float32: (1e-4, 1e-4),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (2e-2, 2e-2),
}


def _plan_kernel_count(plan: AffinePlan, act: str) -> int:
    # Prologue + matvec / wide / full-K, or prologue + split-K + reduce; the
    # fp matvec and wide kernels have no prologue.
    if plan.bn <= 0:
        return 1 if act == "fp" else 2
    return 3 if plan.splits > 1 else 2


def _kernel_count(
    n: int, k: int, gs: int, act: str, m: int, test_m: Optional[int]
) -> int:
    """Static M emits only its plan; dynamic M emits every plan's branch."""
    plans = affine_plans(n, k, gs, act)
    if test_m is not None:
        return sum(_plan_kernel_count(p, act) for p in plans)
    return _plan_kernel_count(next(p for p in plans if m <= p.max_m), act)


class Int4AffineGemmTest(OpTestCase):
    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        act: str,
        group_size: Optional[int],
        dtype: torch.dtype = torch.bfloat16,
        test_m: Optional[int] = None,
        qparam_dtype: torch.dtype = torch.float16,
    ):
        self.m = m
        self.n = n
        self.k = k
        self.act = act
        self.group_size = group_size
        self.dtype = dtype
        self.test_m = test_m
        self.qparam_dtype = qparam_dtype
        dtype_name = str(dtype).removeprefix("torch.")
        dyn = f"_dyn{test_m}" if test_m is not None else ""
        gs_name = f"gs{group_size}" if group_size is not None else "pc"
        qp = "_qpf32" if qparam_dtype == torch.float32 else ""
        self.name = (
            f"{act}_int4_affine_gemm_{gs_name}_m{m}_n{n}_k{k}_{dtype_name}{dyn}{qp}"
        )
        self.rtol, self.atol = _TOLERANCES[dtype]
        self.expected_node_counts = {
            "MetalKernelNode": _kernel_count(n, k, group_size or k, act, m, test_m)
        }

    @classmethod
    def get_test_configs(cls) -> List["Int4AffineGemmTest"]:
        shapes = [
            {"m": 1, "n": 64, "k": 256},
            {"m": 7, "n": 33, "k": 512},
            {"m": 33, "n": 95, "k": 768},
            {"m": 100, "n": 256, "k": 512},
            # Matvec (K % 512 == 0, small M), N not a multiple of 8.
            {"m": 1, "n": 100, "k": 1024},
            {"m": 2, "n": 100, "k": 1024},
            {"m": 1, "n": 1024, "k": 4096},
            # Split-K over padded 8- and 16-row tiles.
            {"m": 5, "n": 96, "k": 1024},
            {"m": 12, "n": 96, "k": 1024, "dtype": torch.float16},
            {"m": 17, "n": 512, "k": 1024, "dtype": torch.float16},
            # Muse Glimmer `o` shape: tuned decode plans for pc / gs256 (fp
            # wide, int8 matvec); gs64 / gs128 take the derived wide plans.
            {"m": 1, "n": 6656, "k": 4096, "dtype": torch.float16},
            {"m": 2, "n": 6656, "k": 4096},
            {"m": 4, "n": 6656, "k": 4096, "dtype": torch.float16},
            # Dynamic M with every plan compiled in.
            {"m": 8, "n": 320, "k": 1024, "test_m": 2},
            {"m": 8, "n": 320, "k": 1024, "test_m": 6},
            {"m": 8, "n": 320, "k": 1024, "test_m": 40},
        ]
        both_acts = [
            # Muse Glimmer `qkv` shape at large M: 64 x 128 tile, bias term
            # after the group loop (BIAS 3).
            {"m": 40, "n": 8704, "k": 6656, "group_size": 128},
            # Uneven split-K: 10 groups in 4 splits of 3, 3, 3, 1 (cooperative
            # 8-row tile at M = 5, 16 x 64 sub-tiles with BIAS 1 at M = 12).
            {"m": 5, "n": 96, "k": 1280, "group_size": 128},
            {"m": 12, "n": 96, "k": 1280, "group_size": 128},
            # fp32 qparam planes: f32 x f32 bias MPP call.
            {
                "m": 33,
                "n": 96,
                "k": 1024,
                "group_size": 128,
                "qparam_dtype": torch.float32,
            },
            {
                "m": 5,
                "n": 96,
                "k": 1024,
                "group_size": None,
                "qparam_dtype": torch.float32,
            },
        ]
        int8_only = [
            {"m": 17, "n": 512, "k": 1024, "dtype": torch.float32, "group_size": 128},
            {"m": 33, "n": 95, "k": 192, "group_size": None},
        ]
        # int8sym shares every GEMM kernel with int8; cover its quantize and
        # epilogue on each lowering (matvec, cooperative split-K, per-SIMD-group
        # tiles, tuned decode, dynamic M).
        sym_shapes = [
            {"m": 1, "n": 100, "k": 1024},
            {"m": 5, "n": 96, "k": 1024},
            {"m": 33, "n": 95, "k": 768},
            {"m": 2, "n": 6656, "k": 4096},
            {"m": 8, "n": 320, "k": 1024, "test_m": 40},
        ]
        configs = [
            cls(**s, act=act, group_size=gs)
            for act in ("int8", "fp")
            for gs in GROUP_SIZES
            for s in shapes
            if gs is None or s["k"] % gs == 0
        ]
        configs += [
            cls(**s, act="int8sym", group_size=gs)
            for gs in GROUP_SIZES
            for s in sym_shapes
            if gs is None or s["k"] % gs == 0
        ]
        configs += [cls(**s, act=act) for act in ACTS for s in both_acts]
        return configs + [cls(**s, act="int8") for s in int8_only]

    def create_model(self) -> nn.Module:
        gs = self.group_size or self.k
        groups = self.k // gs
        wq = torch.randint(-8, 8, (self.n, self.k), dtype=torch.int8)
        w_packed = pack_int4(wq)
        scale = torch.rand(self.n, groups) * 0.02 + 0.005
        zero = torch.rand(self.n, groups) * 4 - 2
        return Int4AffineLinear(w_packed, scale, zero, self.act, self.qparam_dtype)

    def _make_inputs(self, m: int) -> Tuple[torch.Tensor, ...]:
        # A positive offset exercises the activation zero point.
        x = torch.randn(m, self.k, dtype=self.dtype) + 0.5
        # One all-zero row exercises the degenerate range.
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

    parser = argparse.ArgumentParser(
        description="Test mlx::{int8,fp}_int4_affine_gemm ops"
    )
    parser.add_argument("action", choices=["generate", "compare", "run", "list"])
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--filter", type=str, default=None, help="Run configs containing this string"
    )
    args = parser.parse_args()

    if args.rebuild and not rebuild_op_test_runner(verbose=args.verbose):
        sys.exit(1)

    configs = Int4AffineGemmTest.get_test_configs()
    if args.filter:
        configs = [c for c in configs if args.filter in c.name]

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
