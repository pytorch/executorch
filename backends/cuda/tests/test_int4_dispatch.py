#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Int4Tensor F.linear dispatch via int4_dispatch.

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

import unittest

import executorch.backends.cuda.int4_dispatch  # noqa: F401

import torch
import torch.nn as nn
import torch.nn.functional as F
from executorch.examples.models.gemma4_31b.quant.quantize import quantize_weight
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

    module = nn.Linear(K, N, bias=bias, dtype=torch.bfloat16, device="cuda")
    module.weight = nn.Parameter(int4_w.cuda(), requires_grad=False)
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
        up, w_up = _make_int4_linear(512, 256)
        down, w_down = _make_int4_linear(256, 512)
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
        module.weight = nn.Parameter(int4_w, requires_grad=False)
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
        module, w_ref = _make_int4_linear(4096, 5376)
        x = torch.randn(1, 5376, dtype=torch.bfloat16, device="cuda")
        self._check(module(x), F.linear(x, w_ref))

    def test_21504x5376_decode(self):
        module, w_ref = _make_int4_linear(21504, 5376)
        x = torch.randn(1, 5376, dtype=torch.bfloat16, device="cuda")
        self._check(module(x), F.linear(x, w_ref))

    def test_21504x5376_prefill(self):
        module, w_ref = _make_int4_linear(21504, 5376)
        x = torch.randn(128, 5376, dtype=torch.bfloat16, device="cuda")
        self._check(module(x), F.linear(x, w_ref))


if __name__ == "__main__":
    unittest.main()
