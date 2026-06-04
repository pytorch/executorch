#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for ``mlx::gguf_linear`` (GGUF Q6_K linear).

Compares the fused Metal kernels (mat-vec for decode, mat-mat for prefill)
against the eager pure-torch reference on the *same* packed Q6_K weight, so
quantization quality is irrelevant -- only the kernel-vs-reference numerics
are checked. Tolerances follow the activation dtype presets.

``GGUFLinearDynamicTest`` additionally exports once with a symbolic seqlen and
runs the same .pte with M=1 and M>1 to exercise both branches of the runtime
``IfNode`` (decode mat-vec vs prefill mat-mat).

Usage::

    python -m executorch.backends.mlx.custom_kernel_ops.test.test_gguf_linear run
    python -m executorch.backends.mlx.custom_kernel_ops.test.test_gguf_linear run -v
    python -m executorch.backends.mlx.custom_kernel_ops.test.test_gguf_linear run --rebuild
    python -m executorch.backends.mlx.custom_kernel_ops.test.test_gguf_linear eager
"""

from typing import List, Tuple

import executorch.backends.mlx.custom_kernel_ops.gguf_linear  # noqa: F401

import torch
import torch.nn as nn

from executorch.backends.mlx.custom_kernel_ops.gguf_linear import (
    dequantize_q6_k,
    Q6K_BLOCK_BYTES,
    QK_K,
)

from executorch.backends.mlx.test.test_utils import OpTestCase


# ---------------------------------------------------------------------------
# GGUF Q6_K test fixtures.
#
# The Python ``gguf`` package can dequantize Q6_K but does NOT implement Q6_K
# quantization, so we build the packed weight here. Quantization quality is
# irrelevant: the tests only compare the kernel against the eager op on the
# *same* bytes, so we just emit valid random blocks (random ql/qh/scales plus a
# small finite fp16 ``d`` -- the one field that must be finite).
# ---------------------------------------------------------------------------


def make_q6_k_blob(N: int, K: int, seed: int = 0) -> torch.Tensor:
    """Build a ``(N, (K/256)*210)`` uint8 tensor of valid GGUF Q6_K blocks."""
    assert K % QK_K == 0, f"K={K} must be a multiple of {QK_K}"
    nb = K // QK_K
    g = torch.Generator().manual_seed(seed)
    out = torch.empty(N, nb * Q6K_BLOCK_BYTES, dtype=torch.uint8)
    blocks = out.view(N, nb, Q6K_BLOCK_BYTES)
    # ql (0:128) + qh (128:192): any byte values are valid 6-bit quants.
    blocks[..., :192] = torch.randint(
        0, 256, (N, nb, 192), dtype=torch.uint8, generator=g
    )
    # scales (192:208): signed int8 scales (real Q6_K scales can be negative);
    # a modest magnitude keeps dequantized values sane.
    scales = torch.randint(-16, 17, (N, nb, 16), dtype=torch.int32, generator=g)
    blocks[..., 192:208] = scales.to(torch.int8).view(torch.uint8)
    # d (208:210): a small finite fp16 super-block scale. Chosen so dequantized
    # element magnitudes (~ d * scale * (q-32)) are O(0.1), like real Q6_K
    # weights -- the mat-mat kernel stores tiles in half precision (as in
    # llama.cpp), so unrealistically large magnitudes would exceed bf16 tol.
    blocks[..., 208:210] = torch.tensor([7e-4], dtype=torch.float16).view(torch.uint8)
    return out


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class GGUFLinearModel(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.mlx.gguf_linear(x, weight, "q6k", bias)


_DTYPE_TOL = {
    torch.bfloat16: (2e-2, 2e-2),
    # The mat-mat (prefill) kernel stores tiles in half precision (as in
    # llama.cpp), so fp16 outputs are accurate to ~half precision (~4e-3).
    torch.float16: (5e-3, 5e-3),
    torch.float32: (1e-4, 1e-4),
}
_DTYPE_TAG = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}


class GGUFLinearTest(OpTestCase):
    name = "gguf_linear"

    def __init__(
        self,
        M: int = 1,
        N: int = 256,
        K: int = 256,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.M = M
        self.N = N
        self.K = K
        self.dtype = dtype
        self.rtol, self.atol = _DTYPE_TOL[dtype]
        self.name = f"gguf_linear_m{M}_n{N}_k{K}_{_DTYPE_TAG[dtype]}"

    @classmethod
    def get_test_configs(cls) -> List["GGUFLinearTest"]:
        cfgs: List["GGUFLinearTest"] = []
        # Decode (mat-vec).
        for K in (256, 512, 1024):
            for N in (256, 512):
                cfgs.append(cls(M=1, N=N, K=K, dtype=torch.bfloat16))
        cfgs.append(cls(M=1, N=256, K=256, dtype=torch.float16))
        cfgs.append(cls(M=1, N=256, K=256, dtype=torch.float32))
        # Prefill (mat-mat).
        for M in (8, 64, 128):
            cfgs.append(cls(M=M, N=512, K=512, dtype=torch.bfloat16))
        cfgs.append(cls(M=32, N=256, K=256, dtype=torch.float16))
        # Ragged shapes (M and N not multiples of the 32-wide tile / row group).
        cfgs.append(cls(M=40, N=300, K=256, dtype=torch.bfloat16))
        cfgs.append(cls(M=1, N=300, K=256, dtype=torch.bfloat16))
        # Real Gemma-4-31B shapes (hidden=5376, ffn=21504, vocab=262144) to
        # exercise the kernels at production N/K (decode + prefill).
        cfgs.append(cls(M=1, N=4096, K=5376, dtype=torch.bfloat16))  # attn_v
        cfgs.append(cls(M=1, N=5376, K=21504, dtype=torch.bfloat16))  # ffn_down
        cfgs.append(cls(M=8, N=5376, K=21504, dtype=torch.bfloat16))  # ffn_down prefill
        cfgs.append(cls(M=1, N=262144, K=5376, dtype=torch.bfloat16))  # lm_head
        return cfgs

    def create_model(self) -> nn.Module:
        return GGUFLinearModel()

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        torch.manual_seed(0)
        x = torch.randn(self.M, self.K, dtype=self.dtype)
        weight = make_q6_k_blob(self.N, self.K)
        bias = torch.randn(self.N, dtype=self.dtype)
        return (x, weight, bias)


class GGUFLinearDynamicTest(OpTestCase):
    """Dynamic seqlen: export once with a symbolic M, run with M=1 (decode /
    else chain) and M>1 (prefill / then chain) to exercise both IfNode branches.
    """

    name = "gguf_linear_dynamic"

    def __init__(
        self,
        export_M: int = 4,
        test_M: int = 1,
        N: int = 512,
        K: int = 512,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.export_M = export_M
        self.test_M = test_M
        self.seq_len = export_M  # used by create_inputs (export tracing)
        self.N = N
        self.K = K
        self.dtype = dtype
        self.rtol, self.atol = _DTYPE_TOL[dtype]
        self.name = (
            f"gguf_linear_dyn_exp{export_M}_test{test_M}_n{N}_k{K}_"
            f"{_DTYPE_TAG[dtype]}"
        )

    @classmethod
    def get_test_configs(cls) -> List["GGUFLinearDynamicTest"]:
        return [
            cls(export_M=4, test_M=1, dtype=torch.bfloat16),  # decode / else
            cls(export_M=4, test_M=8, dtype=torch.bfloat16),  # prefill / then
            cls(export_M=4, test_M=4, dtype=torch.bfloat16),  # control
            cls(export_M=4, test_M=1, dtype=torch.float16),
            cls(export_M=4, test_M=40, N=300, K=256, dtype=torch.bfloat16),  # ragged
        ]

    def get_dynamic_shapes(self):
        seq_dim = torch.export.Dim("seq_len", min=1, max=64)
        return {"x": {0: seq_dim}, "weight": None, "bias": None}

    def create_model(self) -> nn.Module:
        return GGUFLinearModel()

    def _make_inputs(self, M: int) -> Tuple[torch.Tensor, ...]:
        # Deterministic weight/bias so export-time and run-time (and the eager
        # reference) all use the same quantized weight (it is a runtime input).
        torch.manual_seed(0)
        weight = make_q6_k_blob(self.N, self.K)
        bias = torch.randn(self.N, dtype=self.dtype)
        x = torch.randn(M, self.K, dtype=self.dtype)
        return (x, weight, bias)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return self._make_inputs(self.export_M)

    def create_test_inputs(self) -> Tuple[torch.Tensor, ...]:
        return self._make_inputs(self.test_M)


def _eager_sanity() -> None:
    """Quick CPU check: dequant + matmul matches the eager op on the same bytes."""
    torch.manual_seed(0)
    N, K = 4, 512
    packed = make_q6_k_blob(N, K)
    w_deq = dequantize_q6_k(packed, K)
    print(f"dequant finite: {torch.isfinite(w_deq).all().item()}")
    x = torch.randn(3, K)
    ref = x @ w_deq.t()
    out = torch.ops.mlx.gguf_linear(x, packed, "q6k", None)
    err = (out - ref).abs().max()
    print(f"eager op vs reference max abs err: {err.item():.6e}")
    assert err < 1e-3, err
    # Unsupported format raises.
    try:
        torch.ops.mlx.gguf_linear(x, packed, "q4k", None)
        raise AssertionError("expected NotImplementedError for q4k")
    except RuntimeError as e:
        print(f"q4k correctly rejected: {type(e).__name__}")
    print("eager sanity OK")


if __name__ == "__main__":  # noqa: C901
    import argparse
    import sys

    from executorch.backends.mlx.test.test_utils import rebuild_op_test_runner

    parser = argparse.ArgumentParser(description="Test mlx::gguf_linear op")
    parser.add_argument(
        "action", choices=["generate", "compare", "run", "list", "eager"]
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    if args.action == "eager":
        _eager_sanity()
        sys.exit(0)

    if args.rebuild and not rebuild_op_test_runner(verbose=args.verbose):
        sys.exit(1)

    configs = (
        GGUFLinearTest.get_test_configs() + GGUFLinearDynamicTest.get_test_configs()
    )

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
