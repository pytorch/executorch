#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Int4Tensor F.linear dispatch via quantize_op_dispatch.int4_dispatch.

These tests validate the eager / trace-time dispatch path — the same code
that torch.export traces through when building the AOTI graph. They do NOT
test the .pte runtime C shim (dp4a kernel); that is covered by
test_aoti_torch_cuda_int4_plain_mm.cpp (C++ unit tests) and
test_cuda_pipeline.py::TestCudaExport (end-to-end export + lower).

The API contract: after importing int4_dispatch, F.linear and nn.Linear
with Int4Tensor weights produce numerically correct results. Tests verify
this across decode (M<=4), prefill (M>4), batched (3D), bias, group sizes,
and symmetric/asymmetric quantization. Correctness is measured as mean
relative error against the unquantized bf16 reference (not per-element
atol/rtol, which is too strict for INT4 quantization noise).

Usage:
  python -m pytest backends/cuda/tests/test_int4_dispatch.py -v
"""

import contextlib
import unittest
from unittest import mock

import executorch.backends.cuda.quantize_op_dispatch.int4_dispatch  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as F
from executorch.backends.cuda.coalesced_int4_tensor import CudaCoalescedInt4Tensor
from executorch.backends.cuda.quantize_op_dispatch.int4_dispatch import _dequant_matmul
from executorch.examples.models.gemma4_31b.quant.pack_cuda import pack_linear_for_cuda
from executorch.examples.models.gemma4_31b.quant.quantize import (
    dequantize_weight,
    quantize_weight,
)
from executorch.examples.models.gemma4_31b.quant.recipe import QuantConfig


def _require_cuda(tc: unittest.TestCase) -> None:
    if not torch.cuda.is_available():
        tc.skipTest("CUDA required")


def _make_int4_linear(N, K, group_size=128, symmetric=False, bias=False):
    """Build an nn.Linear with Int4Tensor weight and return (module, bf16_ref_weight).

    The bf16 reference is the original unquantized weight, so tests can
    measure quantization error against the true value.
    """
    w_bf16 = torch.randn(N, K, dtype=torch.bfloat16)
    config = QuantConfig(
        bits=4, group_size=group_size, symmetric=symmetric, method="min_max"
    )
    int4_w = quantize_weight(w_bf16, config)

    # device="cuda" so the random init draws from the CUDA RNG to match the
    # same random weight as regular int4 dispatch and fit the same numerical
    # error tolerance.
    module = nn.Linear(K, N, bias=bias, dtype=torch.bfloat16, device="cuda")
    pack_linear_for_cuda(module, {"weight": int4_w})
    module.cuda()
    return module, w_bf16.cuda()


class TestFLinearDispatch(unittest.TestCase):
    """F.linear with Int4Tensor weight produces correct results."""

    def setUp(self):
        _require_cuda(self)
        torch.manual_seed(42)

    def _check(self, out, ref, tol=0.15):
        rel_err = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
        self.assertLess(rel_err.item(), tol)

    def test_decode_m1(self):
        module, w_ref = _make_int4_linear(256, 512)
        x = torch.randn(1, 512, dtype=torch.bfloat16, device="cuda")
        self._check(module(x), F.linear(x, w_ref))

    def test_prefill_m64(self):
        module, w_ref = _make_int4_linear(256, 512)
        x = torch.randn(64, 512, dtype=torch.bfloat16, device="cuda")
        self._check(module(x), F.linear(x, w_ref))

    def test_3d_batched_input(self):
        module, w_ref = _make_int4_linear(256, 512)
        x = torch.randn(2, 32, 512, dtype=torch.bfloat16, device="cuda")
        out = module(x)
        self.assertEqual(out.shape, (2, 32, 256))
        self._check(out, F.linear(x, w_ref))

    def test_with_bias(self):
        module, w_ref = _make_int4_linear(256, 512, bias=True)
        x = torch.randn(4, 512, dtype=torch.bfloat16, device="cuda")
        self._check(module(x), F.linear(x, w_ref, module.bias))

    def test_group_size_32(self):
        module, w_ref = _make_int4_linear(128, 256, group_size=32)
        x = torch.randn(1, 256, dtype=torch.bfloat16, device="cuda")
        self._check(module(x), F.linear(x, w_ref))

    def test_symmetric(self):
        module, w_ref = _make_int4_linear(256, 512, symmetric=True)
        x = torch.randn(1, 512, dtype=torch.bfloat16, device="cuda")
        self._check(module(x), F.linear(x, w_ref))


class TestMultiLayer(unittest.TestCase):
    """Dispatch works across multiple Int4 linear modules in a model."""

    def setUp(self):
        _require_cuda(self)
        torch.manual_seed(42)

    def _check(self, out, ref, tol=0.15):
        rel_err = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
        self.assertLess(rel_err.item(), tol)

    def test_two_layer_mlp(self):
        up, w_up = _make_int4_linear(512, 256, group_size=32)
        down, w_down = _make_int4_linear(256, 512, group_size=32)
        x = torch.randn(4, 256, dtype=torch.bfloat16, device="cuda")
        out = down(F.silu(up(x)))
        ref = F.linear(F.silu(F.linear(x, w_up)), w_down)
        self._check(out, ref)

    def test_sequential_decode_steps(self):
        module, w_ref = _make_int4_linear(256, 512)
        for _ in range(4):
            x = torch.randn(1, 512, dtype=torch.bfloat16, device="cuda")
            self._check(module(x), F.linear(x, w_ref))


class TestCompile(unittest.TestCase):
    """Dispatch works under torch.compile."""

    def setUp(self):
        _require_cuda(self)
        torch.manual_seed(42)

    def _check(self, out, ref, tol=0.15):
        rel_err = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
        self.assertLess(rel_err.item(), tol)

    def test_compile_decode(self):
        module, w_ref = _make_int4_linear(256, 512)
        compiled = torch.compile(module, fullgraph=True)
        x = torch.randn(1, 512, dtype=torch.bfloat16, device="cuda")
        self._check(compiled(x), F.linear(x, w_ref))

    def test_compile_prefill(self):
        module, w_ref = _make_int4_linear(256, 512)
        compiled = torch.compile(module, fullgraph=True)
        x = torch.randn(64, 512, dtype=torch.bfloat16, device="cuda")
        self._check(compiled(x), F.linear(x, w_ref))

    def test_compile_matches_eager(self):
        module, _ = _make_int4_linear(256, 512)
        compiled = torch.compile(module, fullgraph=True)
        x = torch.randn(4, 512, dtype=torch.bfloat16, device="cuda")
        out_eager = module(x)
        out_compiled = compiled(x)
        self.assertTrue(torch.allclose(out_eager, out_compiled, atol=0.5))


class TestDeviceMovement(unittest.TestCase):
    """Int4Tensor weight survives device movement and still dispatches."""

    def setUp(self):
        _require_cuda(self)
        torch.manual_seed(42)

    def _check(self, out, ref, tol=0.15):
        rel_err = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
        self.assertLess(rel_err.item(), tol)

    def test_to_cuda(self):
        w_bf16 = torch.randn(256, 512, dtype=torch.bfloat16)
        config = QuantConfig(bits=4, group_size=128, symmetric=False, method="min_max")
        int4_w = quantize_weight(w_bf16, config)
        module = nn.Linear(512, 256, bias=False)
        pack_linear_for_cuda(module, {"weight": int4_w})
        module = module.to("cuda")
        x = torch.randn(1, 512, dtype=torch.bfloat16, device="cuda")
        self._check(module(x), F.linear(x, w_bf16.cuda()))


class TestLargeShapes(unittest.TestCase):
    """Correctness at large production-scale layer shapes."""

    def setUp(self):
        _require_cuda(self)
        torch.manual_seed(42)

    def _check(self, out, ref, tol=0.15):
        rel_err = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
        self.assertLess(rel_err.item(), tol)

    def test_4096x5376_decode(self):
        module, w_ref = _make_int4_linear(4096, 5376, group_size=32)
        x = torch.randn(1, 5376, dtype=torch.bfloat16, device="cuda")
        self._check(module(x), F.linear(x, w_ref))

    def test_21504x5376_decode(self):
        module, w_ref = _make_int4_linear(21504, 5376, group_size=32)
        x = torch.randn(1, 5376, dtype=torch.bfloat16, device="cuda")
        self._check(module(x), F.linear(x, w_ref))

    def test_21504x5376_prefill(self):
        module, w_ref = _make_int4_linear(21504, 5376, group_size=32)
        x = torch.randn(128, 5376, dtype=torch.bfloat16, device="cuda")
        self._check(module(x), F.linear(x, w_ref))


def _make_int4_tensor(N, K, group_size=128, symmetric=False):
    """Build a stock torchao ``Int4Tensor`` (NOT packed/coalesced) on CPU."""
    w = torch.randn(N, K, dtype=torch.bfloat16)
    config = QuantConfig(
        bits=4, group_size=group_size, symmetric=symmetric, method="min_max"
    )
    return quantize_weight(w, config), w


@contextlib.contextmanager
def _record_int4_plain_mm():
    """Record calls to the decode custom op without needing a GPU.

    Replaces ``torch.ops.executorch_cuda.int4_plain_mm`` (whose real impl is the
    CUDA C shim) with a recorder that computes the result via the eager CPU
    dequant, so the dispatch handler still returns a valid tensor.
    """
    calls = []

    def _fake(self, qdata, scale, scale_step, zero, zero_step, group_size):
        calls.append((tuple(self.shape), group_size))
        return _dequant_matmul(
            self, qdata, scale, scale_step, zero, zero_step, group_size
        )

    with mock.patch.object(torch.ops.executorch_cuda, "int4_plain_mm", _fake):
        yield calls


class TestDispatchRouting(unittest.TestCase):
    """Type-based routing: only CudaCoalescedInt4Tensor reaches int4_plain_mm.

    These tests run without a GPU by recording calls to the decode custom op
    and computing the result with the eager CPU dequant. They guard the
    comment-8 refactor: the CUDA decode path must be selected by weight *type*,
    not by globally overriding torchao ``Int4Tensor``'s F.linear.
    """

    def setUp(self):
        torch.manual_seed(0)

    def _rel_err(self, out, ref):
        return (
            (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
        ).item()

    def test_stock_int4tensor_does_not_route_to_int4_plain_mm(self):
        """A plain torchao Int4Tensor must fall back to torchao's default path."""
        t, _ = _make_int4_tensor(16, 64, group_size=32)
        x = torch.randn(1, 64, dtype=torch.bfloat16)  # M=1 (decode regime)
        with _record_int4_plain_mm() as calls:
            # torchao's default path uses mslk/CUDA and is not exercised on CPU;
            # we only assert that our decode op is NOT reached.
            with contextlib.suppress(Exception):
                F.linear(x, t)
        self.assertEqual(calls, [])

    def test_coalesced_tensor_routes_to_int4_plain_mm(self):
        """CudaCoalescedInt4Tensor with M<=4 routes to the decode custom op."""
        t, _ = _make_int4_tensor(16, 256, group_size=32)
        c = CudaCoalescedInt4Tensor.from_int4_tensor(t)
        x = torch.randn(1, 256, dtype=torch.bfloat16)  # M=1 (decode regime)
        with _record_int4_plain_mm() as calls:
            out = F.linear(x, c)
        self.assertEqual(len(calls), 1)
        self.assertEqual(out.shape, (1, 16))

    def test_coalesced_tensor_prefill_uses_dequant(self):
        """M>4 uses inline dequant (no custom op) and is numerically correct."""
        t, _ = _make_int4_tensor(16, 256, group_size=32)
        c = CudaCoalescedInt4Tensor.from_int4_tensor(t)
        x = torch.randn(8, 256, dtype=torch.bfloat16)  # M=8 > 4 (prefill regime)
        with _record_int4_plain_mm() as calls:
            out = F.linear(x, c)
        self.assertEqual(calls, [])
        ref = F.linear(x, dequantize_weight(t, torch.bfloat16))
        self.assertLess(self._rel_err(out, ref), 0.02)

    def test_square_shape_not_misrouted(self):
        """N == n_groups (square scale) stock tensor is still not routed.

        K = group_size * N makes scale square (n_groups == N); the old shape
        heuristic could not distinguish this coalesced-looking case. Type-based
        routing makes the scale shape irrelevant.
        """
        t, _ = _make_int4_tensor(4, 128, group_size=32)
        self.assertEqual(tuple(t.scale.shape), (4, 4))  # (n_groups, N), square
        x = torch.randn(1, 128, dtype=torch.bfloat16)
        with _record_int4_plain_mm() as calls:
            with contextlib.suppress(Exception):
                F.linear(x, t)
        self.assertEqual(calls, [])

    def test_from_int4_tensor_transpose_correct(self):
        """from_int4_tensor owns the (n_groups, N) -> (N, n_groups) transpose."""
        t, _ = _make_int4_tensor(24, 256, group_size=64)
        c = CudaCoalescedInt4Tensor.from_int4_tensor(t)
        n_groups = 256 // 64
        self.assertEqual(tuple(t.scale.shape), (n_groups, 24))  # torchao layout
        self.assertEqual(tuple(c.scale.shape), (24, n_groups))  # coalesced layout
        # Scale is a uint8 code + a per-256 fp16 step; zero is a uint8 code + a
        # per-row bf16 step. Decoding must recover the transposed torchao
        # scale/zero (within code quant error).
        n_super = int(c.scale_step.shape[1])
        gps = n_groups // n_super
        scale_step_g = c.scale_step.to(torch.bfloat16).repeat_interleave(gps, dim=1)
        dec_scale = c.scale.to(torch.bfloat16) * scale_step_g
        dec_zero = c.zero_point.to(torch.bfloat16) * c.zero_step.to(torch.bfloat16)
        torch.testing.assert_close(
            dec_scale, t.scale.t().contiguous().to(torch.bfloat16), rtol=0.05, atol=0
        )
        torch.testing.assert_close(
            dec_zero,
            t.zero_point.t().contiguous().to(torch.bfloat16),
            rtol=0.02,
            atol=0,
        )
        # End-to-end decode result matches a reference dequant of the original.
        x = torch.randn(2, 256, dtype=torch.bfloat16)
        with _record_int4_plain_mm() as calls:
            out = F.linear(x, c)
        self.assertEqual(len(calls), 1)
        ref = F.linear(x, dequantize_weight(t, torch.bfloat16))
        self.assertLess(self._rel_err(out, ref), 0.02)


if __name__ == "__main__":
    unittest.main()
