#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for CudaDp4aPlanarInt5Tensor F.linear dispatch via int5_dispatch.

These tests validate the eager / trace-time dispatch path — the same code that
torch.export traces through when building the AOTI graph. They do NOT test the
.pte runtime C shim (W5A8 dp4a kernel); that is covered by
test_aoti_torch_cuda_int5_plain_mm.cpp (C++ unit tests).

The API contract: after importing int5_dispatch, F.linear / nn.Linear with a
CudaDp4aPlanarInt5Tensor weight produce numerically correct results, routed by
batch size (decode M<=4 -> custom op, prefill M>4 -> inline dequant). Q5_K is
asymmetric (has a zero point, like INT4): ``w = scale * (u - zero)`` with u in
[0, 31]. Both scale and zero are stored as per-group uint8 codes with a per-256
fp16 step packed into ONE warp-shuffle word by the kernel (z_pack). Routing tests
run without a GPU by recording calls to the decode custom op.

Usage:
  python -m pytest backends/cuda/tests/test_int5_dispatch.py -v
"""

import contextlib
import unittest
from unittest import mock

import executorch.backends.cuda.quantize_op_dispatch.int5_dispatch  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as F
from executorch.backends.cuda.dp4a_planar_int5_tensor import (
    _encode_uint8_per_super,
    CudaDp4aPlanarInt5Tensor,
    pack_int5,
    unpack_int5,
)
from executorch.backends.cuda.quantize_op_dispatch.int5_dispatch import (
    _dequant_matmul_int5,
)

GS = 32  # GGUF Q5_K group size


def _require_cuda(tc: unittest.TestCase) -> None:
    if not torch.cuda.is_available():
        tc.skipTest("CUDA required")


class _FakeIntx:
    """Minimal stand-in for IntxUnpackedToInt8Tensor (asymmetric int5)."""


def _make_intx_source(N, K, gs=GS):
    """Build an exact Q5_K-affine int5 source: centered qdata [-16,15], zero>=0.

    Returns (fake_intx, u[N,K] uint8, eff_scale[N,ng] f32, zero[N,ng] f32) where
    the exact reference is ``w = eff_scale * (u - zero)``.
    """
    ng = K // gs
    w = torch.randn(N, K, dtype=torch.float32) * (0.5 + torch.rand(N, 1))
    wg = w.reshape(N, ng, gs)
    wmin = wg.amin(dim=2)
    wmax = wg.amax(dim=2)
    eff_scale = ((wmax - wmin) / 31.0).clamp_min(1e-6)
    u = torch.round((wg - wmin.unsqueeze(-1)) / eff_scale.unsqueeze(-1)).clamp_(0, 31)
    u = u.to(torch.int16).reshape(N, K)
    zero = (-wmin / eff_scale).clamp_min(0.0)  # (N, ng) >= 0
    ft = _FakeIntx()
    ft.qdata = (u - 16).to(torch.int8)  # centered [-16, 15]
    ft.zero_point = (zero - 16.0).to(torch.bfloat16)  # centered like gguf.py
    ft.scale = eff_scale.to(torch.bfloat16)
    ft.block_size = [1, gs]
    ft.shape = torch.Size([N, K])
    return ft, u.to(torch.uint8), eff_scale, zero


def _make_int5_tensor(N, K, gs=GS):
    """Build a CudaDp4aPlanarInt5Tensor and return (tensor, eff reference weight).

    The effective reference is the tensor's OWN dequant (decodes the per-256
    z_pack codes/steps), so tests measure the kernel/dispatch against the exact
    metadata the kernel reads — not the pre-quant float scale/zero.
    """
    ft, _, _, _ = _make_intx_source(N, K, gs)
    t = CudaDp4aPlanarInt5Tensor._from_intx_int8(ft)
    w_ref = t.dequantize(torch.bfloat16)
    return t, w_ref


@contextlib.contextmanager
def _record_int5_plain_mm():
    """Record calls to the decode custom op without needing a GPU.

    Replaces ``torch.ops.executorch_cuda.int5_plain_mm`` (whose real impl is the
    CUDA C shim) with a recorder that computes the result via the eager CPU
    dequant, so the dispatch handler still returns a valid tensor.
    """
    calls = []

    def _fake(self, ql, qh, scale, scale_step, zero, zero_step, group_size):
        calls.append((tuple(self.shape), group_size))
        return _dequant_matmul_int5(
            self, ql, qh, scale, scale_step, zero, zero_step, group_size
        )

    with mock.patch.object(torch.ops.executorch_cuda, "int5_plain_mm", _fake):
        yield calls


class TestPacker(unittest.TestCase):
    """pack/unpack round-trip and per-256 metadata encoding are self-consistent."""

    def setUp(self):
        torch.manual_seed(0)

    def test_pack_unpack_bit_exact(self):
        u = torch.randint(0, 32, (8, 512), dtype=torch.uint8)
        ql, qh = pack_int5(u)
        self.assertEqual(tuple(ql.shape), (8, 256))
        self.assertEqual(tuple(qh.shape), (8, 64))
        self.assertTrue(torch.equal(unpack_int5(ql, qh, 8, 512), u))

    def test_encode_uint8_per_super_shapes(self):
        N, K = 4, 1024
        ng = K // GS
        n_super = K // 256
        x = torch.rand(N, ng).abs() + 0.01
        codes, step = _encode_uint8_per_super(x, GS)
        self.assertEqual(tuple(codes.shape), (N, ng))
        self.assertEqual(tuple(step.shape), (N, n_super))
        self.assertEqual(step.dtype, torch.float16)
        self.assertEqual(codes.dtype, torch.uint8)

    def test_tensor_fields(self):
        t, _ = _make_int5_tensor(16, 512)
        self.assertEqual(
            t.tensor_data_names,
            ["ql", "qh", "scale", "scale_step", "zero_point", "zero_step"],
        )
        self.assertEqual(t.scale_step.dtype, torch.float16)
        self.assertEqual(t.zero_step.dtype, torch.float16)
        self.assertEqual(tuple(t.scale_step.shape), (16, 512 // 256))


class TestDispatchRouting(unittest.TestCase):
    """Type-based routing: M<=4 -> int5_plain_mm op, M>4 -> inline dequant.

    Runs without a GPU by recording calls to the decode custom op and computing
    the result with the eager CPU dequant.
    """

    def setUp(self):
        torch.manual_seed(0)

    def _rel_err(self, out, ref):
        return (
            (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
        ).item()

    def test_decode_routes_to_int5_plain_mm(self):
        t, _ = _make_int5_tensor(16, 256)
        x = torch.randn(1, 256, dtype=torch.bfloat16)  # M=1 (decode regime)
        with _record_int5_plain_mm() as calls:
            out = F.linear(x, t)
        self.assertEqual(len(calls), 1)
        self.assertEqual(out.shape, (1, 16))

    def test_prefill_uses_dequant(self):
        t, w_ref = _make_int5_tensor(16, 256)
        x = torch.randn(8, 256, dtype=torch.bfloat16)  # M=8 > 4 (prefill regime)
        with _record_int5_plain_mm() as calls:
            out = F.linear(x, t)
        self.assertEqual(calls, [])
        ref = F.linear(x, w_ref)
        self.assertLess(self._rel_err(out, ref), 0.02)

    def test_decode_result_matches_reference(self):
        t, w_ref = _make_int5_tensor(24, 512)
        x = torch.randn(2, 512, dtype=torch.bfloat16)
        with _record_int5_plain_mm() as calls:
            out = F.linear(x, t)
        self.assertEqual(len(calls), 1)
        ref = F.linear(x, w_ref)
        self.assertLess(self._rel_err(out, ref), 0.02)

    def test_with_bias(self):
        t, w_ref = _make_int5_tensor(16, 256)
        bias = torch.randn(16, dtype=torch.bfloat16)
        x = torch.randn(1, 256, dtype=torch.bfloat16)
        with _record_int5_plain_mm():
            out = F.linear(x, t, bias)
        ref = F.linear(x, w_ref, bias)
        self.assertLess(self._rel_err(out, ref), 0.02)

    def test_with_bias_kwarg(self):
        t, w_ref = _make_int5_tensor(16, 256)
        bias = torch.randn(16, dtype=torch.bfloat16)
        x = torch.randn(1, 256, dtype=torch.bfloat16)
        with _record_int5_plain_mm():
            out = F.linear(x, t, bias=bias)
        ref = F.linear(x, w_ref, bias)
        self.assertLess(self._rel_err(out, ref), 0.02)
        with _record_int5_plain_mm():
            out_no_bias = F.linear(x, t)
        self.assertTrue(
            torch.allclose(out, out_no_bias + bias, atol=1e-2),
            "keyword bias was not applied",
        )

    def test_3d_batched_input(self):
        t, w_ref = _make_int5_tensor(16, 256)
        x = torch.randn(2, 8, 256, dtype=torch.bfloat16)  # flattened M=16 > 4
        with _record_int5_plain_mm() as calls:
            out = F.linear(x, t)
        self.assertEqual(calls, [])  # prefill regime
        self.assertEqual(out.shape, (2, 8, 16))
        ref = F.linear(x, w_ref)
        self.assertLess(self._rel_err(out, ref), 0.02)

    def test_from_intx_int8_roundtrip(self):
        """_from_intx_int8 packs an asymmetric int5 tensor and dispatch is correct."""
        ft, u, _, _ = _make_intx_source(16, 256)
        t = CudaDp4aPlanarInt5Tensor._from_intx_int8(ft)
        # Weight bit-pack is lossless: unpacked u matches the source.
        self.assertTrue(torch.equal(unpack_int5(t.ql, t.qh, 16, 256), u))
        x = torch.randn(1, 256, dtype=torch.bfloat16)
        with _record_int5_plain_mm() as calls:
            out = F.linear(x, t)
        self.assertEqual(len(calls), 1)
        # Reference is the tensor's own fp32-accumulated dequant; the op path
        # accumulates in bf16, so ~1% bf16 rounding is expected (not an error).
        ref = F.linear(x, t.dequantize(torch.bfloat16))
        self.assertLess(self._rel_err(out, ref), 0.02)

    def test_from_intx_int8_rejects_out_of_range(self):
        """qdata outside [-16, 15] (not a genuine int5) is rejected."""
        ft, _, _, _ = _make_intx_source(8, 64)
        ft.qdata = ft.qdata.clone()
        ft.qdata[0, 0] = 20  # > 15
        with self.assertRaises(ValueError):
            CudaDp4aPlanarInt5Tensor._from_intx_int8(ft)

    def test_from_exportable_gguf(self):
        """from_exportable_gguf reuses the gguf.py Q5_K decode then packs the planes."""
        from executorch.extension.llm.export.gguf import (
            _Q5_K_BLOCK_BYTES,
            ExportableGGUFTensor,
        )

        N, nb = 8, 1  # K = nb * 256
        g = torch.Generator().manual_seed(0)
        blk = torch.randint(
            0, 256, (N * nb, _Q5_K_BLOCK_BYTES), dtype=torch.uint8, generator=g
        )
        # Fixed non-zero fp16 d / dmin super-scales so decode is well-conditioned.
        blk[:, 0:2] = torch.tensor([0.02], dtype=torch.float16).view(torch.uint8)
        blk[:, 2:4] = torch.tensor([0.01], dtype=torch.float16).view(torch.uint8)
        raw = blk.reshape(N, nb * _Q5_K_BLOCK_BYTES)
        gt = ExportableGGUFTensor.from_raw(raw, "q5_k")

        t = CudaDp4aPlanarInt5Tensor.from_exportable_gguf(gt)
        self.assertIsInstance(t, CudaDp4aPlanarInt5Tensor)
        self.assertEqual(tuple(t.shape), (N, nb * 256))
        # The weight bit-pack reuses the shared Q5_K decode losslessly: unpacked
        # u matches the centered qdata (+16) from the int8 path.
        intx = gt.to_intx_unpacked_to_int8_tensor()
        u_rt = unpack_int5(t.ql, t.qh, N, nb * 256).to(torch.int16)
        self.assertTrue(torch.equal(u_rt, intx.qdata.to(torch.int16) + 16))

    def test_from_exportable_gguf_rejects_non_q5k(self):
        """A non-q5_k ExportableGGUFTensor is rejected before any decode."""
        from executorch.extension.llm.export.gguf import (
            _Q4_K_BLOCK_BYTES,
            ExportableGGUFTensor,
        )

        raw = torch.zeros(4, _Q4_K_BLOCK_BYTES, dtype=torch.uint8)
        gt = ExportableGGUFTensor.from_raw(raw, "q4_k")
        with self.assertRaises(ValueError):
            CudaDp4aPlanarInt5Tensor.from_exportable_gguf(gt)


class TestFLinearDispatchCuda(unittest.TestCase):
    """F.linear with a CudaDp4aPlanarInt5Tensor weight on CUDA (eager -> dequant)."""

    def setUp(self):
        _require_cuda(self)
        torch.manual_seed(42)

    def _check(self, out, ref, tol=0.02):
        rel_err = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
        self.assertLess(rel_err.item(), tol)

    def _linear(self, N, K, gs=GS):
        t, w_ref = _make_int5_tensor(N, K, gs)
        module = nn.Linear(K, N, bias=False, dtype=torch.bfloat16)
        module.weight = nn.Parameter(t, requires_grad=False)
        module.cuda()
        return module, w_ref.cuda()

    def test_decode_m1(self):
        module, w_ref = self._linear(256, 512)
        x = torch.randn(1, 512, dtype=torch.bfloat16, device="cuda")
        self._check(module(x), F.linear(x, w_ref))

    def test_prefill_m64(self):
        module, w_ref = self._linear(256, 512)
        x = torch.randn(64, 512, dtype=torch.bfloat16, device="cuda")
        self._check(module(x), F.linear(x, w_ref))

    def test_dequantize_matches_reference(self):
        t, w_ref = _make_int5_tensor(32, 256)
        # Explicit dtype: scale/zero are uint8 codes, so the default would
        # dequantize in an integer dtype and collapse to 0.
        self.assertTrue(torch.equal(t.dequantize(torch.bfloat16).cpu(), w_ref.cpu()))


if __name__ == "__main__":
    unittest.main()
