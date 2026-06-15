#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for CudaDp4aPlanarInt6Tensor F.linear dispatch via int6_dispatch.

These tests validate the eager / trace-time dispatch path — the same code that
torch.export traces through when building the AOTI graph. They do NOT test the
.pte runtime C shim (W6A8 dp4a kernel); that is covered by
test_aoti_torch_cuda_int6_plain_mm.cpp (C++ unit tests).

The API contract: after importing int6_dispatch, F.linear / nn.Linear with a
CudaDp4aPlanarInt6Tensor weight produce numerically correct results, routed by
batch size (decode M<=4 -> custom op, prefill M>4 -> inline dequant). Routing
tests run without a GPU by recording calls to the decode custom op.

Usage:
  python -m pytest backends/cuda/tests/test_int6_dispatch.py -v
"""

import contextlib
import unittest
from unittest import mock

import executorch.backends.cuda.quantize_op_dispatch.int6_dispatch  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as F
from executorch.backends.cuda.dp4a_planar_int6_tensor import (
    CudaDp4aPlanarInt6Tensor,
    pack_int6,
    unpack_int6,
)
from executorch.backends.cuda.quantize_op_dispatch.int6_dispatch import (
    _dequant_matmul_int6,
)


def _require_cuda(tc: unittest.TestCase) -> None:
    if not torch.cuda.is_available():
        tc.skipTest("CUDA required")


def _make_int6_tensor(N, K, group_size=16):
    """Build a CudaDp4aPlanarInt6Tensor (symmetric Q6_K) and return (tensor, q, scale).

    ``q`` (int8 in [-32, 31]) and ``scale`` are the originals, so tests can
    measure against the exact dequant reference ``w = q * scale``.
    """
    q = torch.randint(-32, 32, (N, K), dtype=torch.int8)
    scale = (torch.rand(N, K // group_size) * 0.1 + 0.01).to(torch.bfloat16)
    ql, qh = pack_int6(q)
    t = CudaDp4aPlanarInt6Tensor(ql, qh, scale, [1, group_size], torch.Size([N, K]))
    return t, q, scale


def _ref_weight(q, scale, group_size, dtype=torch.bfloat16):
    """Exact dequant reference: w[n, k] = q[n, k] * scale[n, k//gs]."""
    N, K = q.shape
    ng = K // group_size
    w = q.to(dtype).reshape(N, ng, group_size) * scale.to(dtype).reshape(N, ng, 1)
    return w.reshape(N, K)


@contextlib.contextmanager
def _record_int6_plain_mm():
    """Record calls to the decode custom op without needing a GPU.

    Replaces ``torch.ops.executorch_cuda.int6_plain_mm`` (whose real impl is the
    CUDA C shim) with a recorder that computes the result via the eager CPU
    dequant, so the dispatch handler still returns a valid tensor.
    """
    calls = []

    def _fake(self, ql, qh, scale, group_size):
        calls.append((tuple(self.shape), group_size))
        return _dequant_matmul_int6(self, ql, qh, scale, group_size)

    with mock.patch.object(torch.ops.executorch_cuda, "int6_plain_mm", _fake):
        yield calls


class TestDispatchRouting(unittest.TestCase):
    """Type-based routing: M<=4 -> int6_plain_mm op, M>4 -> inline dequant.

    Runs without a GPU by recording calls to the decode custom op and computing
    the result with the eager CPU dequant.
    """

    def setUp(self):
        torch.manual_seed(0)

    def _rel_err(self, out, ref):
        return (
            (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
        ).item()

    def test_decode_routes_to_int6_plain_mm(self):
        """M<=4 routes to the decode custom op."""
        t, _, _ = _make_int6_tensor(16, 64)
        x = torch.randn(1, 64, dtype=torch.bfloat16)  # M=1 (decode regime)
        with _record_int6_plain_mm() as calls:
            out = F.linear(x, t)
        self.assertEqual(len(calls), 1)
        self.assertEqual(out.shape, (1, 16))

    def test_prefill_uses_dequant(self):
        """M>4 uses inline dequant (no custom op) and is numerically correct."""
        t, q, scale = _make_int6_tensor(16, 64)
        x = torch.randn(8, 64, dtype=torch.bfloat16)  # M=8 > 4 (prefill regime)
        with _record_int6_plain_mm() as calls:
            out = F.linear(x, t)
        self.assertEqual(calls, [])
        ref = F.linear(x, _ref_weight(q, scale, 16))
        self.assertLess(self._rel_err(out, ref), 0.02)

    def test_decode_result_matches_reference(self):
        """The decode op (eager -> dequant) is numerically correct."""
        t, q, scale = _make_int6_tensor(24, 128)
        x = torch.randn(2, 128, dtype=torch.bfloat16)
        with _record_int6_plain_mm() as calls:
            out = F.linear(x, t)
        self.assertEqual(len(calls), 1)
        ref = F.linear(x, _ref_weight(q, scale, 16))
        self.assertLess(self._rel_err(out, ref), 0.02)

    def test_with_bias(self):
        """Bias is added after the matmul on the decode path."""
        t, q, scale = _make_int6_tensor(16, 64)
        bias = torch.randn(16, dtype=torch.bfloat16)
        x = torch.randn(1, 64, dtype=torch.bfloat16)
        with _record_int6_plain_mm():
            out = F.linear(x, t, bias)
        ref = F.linear(x, _ref_weight(q, scale, 16), bias)
        self.assertLess(self._rel_err(out, ref), 0.02)

    def test_3d_batched_input(self):
        """3D input is flattened and the output shape is restored."""
        t, q, scale = _make_int6_tensor(16, 64)
        x = torch.randn(2, 8, 64, dtype=torch.bfloat16)  # flattened M=16 > 4
        with _record_int6_plain_mm() as calls:
            out = F.linear(x, t)
        self.assertEqual(calls, [])  # prefill regime
        self.assertEqual(out.shape, (2, 8, 16))
        ref = F.linear(x, _ref_weight(q, scale, 16))
        self.assertLess(self._rel_err(out, ref), 0.02)

    def test_from_intx_int8_roundtrip(self):
        """from_intx_int8 packs a symmetric int8 tensor and dispatch is correct."""
        from torchao.quantization import IntxUnpackedToInt8Tensor

        N, K, gs = 16, 64, 16
        q = torch.randint(-32, 32, (N, K), dtype=torch.int8)
        scale = (torch.rand(N, K // gs) * 0.1 + 0.01).to(torch.bfloat16)
        intx = IntxUnpackedToInt8Tensor(
            qdata=q,
            scale=scale,
            zero_point=torch.zeros_like(scale, dtype=torch.int8),
            target_dtype=torch.int8,
            block_size=(1, gs),
            dtype=torch.bfloat16,
            activation_quantization=None,
        )
        t = CudaDp4aPlanarInt6Tensor.from_intx_int8(intx)
        x = torch.randn(1, K, dtype=torch.bfloat16)
        with _record_int6_plain_mm() as calls:
            out = F.linear(x, t)
        self.assertEqual(len(calls), 1)
        ref = F.linear(x, _ref_weight(q, scale, gs))
        self.assertLess(self._rel_err(out, ref), 0.02)

    def test_from_intx_int8_rejects_asymmetric(self):
        """A non-zero zero_point (not Q6_K) is rejected."""
        from torchao.quantization import IntxUnpackedToInt8Tensor

        N, K, gs = 8, 64, 16
        q = torch.randint(-32, 32, (N, K), dtype=torch.int8)
        scale = (torch.rand(N, K // gs) * 0.1 + 0.01).to(torch.bfloat16)
        intx = IntxUnpackedToInt8Tensor(
            qdata=q,
            scale=scale,
            zero_point=torch.ones_like(scale, dtype=torch.int8),
            target_dtype=torch.int8,
            block_size=(1, gs),
            dtype=torch.bfloat16,
            activation_quantization=None,
        )
        with self.assertRaises(ValueError):
            CudaDp4aPlanarInt6Tensor.from_intx_int8(intx)

    def test_from_exportable_gguf(self):
        """from_exportable_gguf reuses the gguf.py Q6_K decode then packs losslessly."""
        from executorch.extension.llm.export.gguf import (
            _Q6_K_BLOCK_BYTES,
            ExportableGGUFTensor,
        )

        N, nb = 8, 1  # K = nb * 256
        g = torch.Generator().manual_seed(0)
        blk = torch.randint(
            0, 256, (N * nb, _Q6_K_BLOCK_BYTES), dtype=torch.uint8, generator=g
        )
        blk[:, 192:208] = 0x10  # fixed non-zero int8 sub-scales
        blk[:, 208:210] = torch.tensor([0.01], dtype=torch.float16).view(
            torch.uint8
        )  # super-block scale d
        raw = blk.reshape(N, nb * _Q6_K_BLOCK_BYTES)
        gt = ExportableGGUFTensor.from_raw(raw, "q6_k")

        t = CudaDp4aPlanarInt6Tensor.from_exportable_gguf(gt)
        self.assertIsInstance(t, CudaDp4aPlanarInt6Tensor)
        self.assertEqual(tuple(t.shape), (N, nb * 256))

        # The packer must reuse the shared Q6_K int8 decode (no duplication) and
        # bit-pack it losslessly: the unpacked q and the scale match the int8 path.
        intx = gt.to_intx_unpacked_to_int8_tensor()
        q_rt = unpack_int6(t.ql, t.qh, N, nb * 256).to(torch.int8)
        self.assertTrue(torch.equal(q_rt, intx.qdata))
        self.assertTrue(torch.equal(t.scale, intx.scale))

    def test_from_exportable_gguf_rejects_non_q6k(self):
        """A non-q6_k ExportableGGUFTensor is rejected before any decode."""
        from executorch.extension.llm.export.gguf import (
            _Q4_K_BLOCK_BYTES,
            ExportableGGUFTensor,
        )

        raw = torch.zeros(4, _Q4_K_BLOCK_BYTES, dtype=torch.uint8)
        gt = ExportableGGUFTensor.from_raw(raw, "q4_k")
        with self.assertRaises(ValueError):
            CudaDp4aPlanarInt6Tensor.from_exportable_gguf(gt)


class TestFLinearDispatchCuda(unittest.TestCase):
    """F.linear with a CudaDp4aPlanarInt6Tensor weight on CUDA (eager -> dequant)."""

    def setUp(self):
        _require_cuda(self)
        torch.manual_seed(42)

    def _check(self, out, ref, tol=0.02):
        rel_err = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
        self.assertLess(rel_err.item(), tol)

    def _linear(self, N, K, gs=16):
        t, q, scale = _make_int6_tensor(N, K, gs)
        module = nn.Linear(K, N, bias=False, dtype=torch.bfloat16)
        module.weight = nn.Parameter(t, requires_grad=False)
        module.cuda()
        return module, _ref_weight(q, scale, gs).cuda()

    def test_decode_m1(self):
        module, w_ref = self._linear(256, 512)
        x = torch.randn(1, 512, dtype=torch.bfloat16, device="cuda")
        self._check(module(x), F.linear(x, w_ref))

    def test_prefill_m64(self):
        module, w_ref = self._linear(256, 512)
        x = torch.randn(64, 512, dtype=torch.bfloat16, device="cuda")
        self._check(module(x), F.linear(x, w_ref))

    def test_dequantize_matches_reference(self):
        t, q, scale = _make_int6_tensor(32, 128)
        ref = _ref_weight(q, scale, 16)
        self.assertTrue(torch.equal(t.dequantize().cpu(), ref))


if __name__ == "__main__":
    unittest.main()
