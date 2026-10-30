#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for CudaDp4aPlanarInt6Tensor F.linear dispatch via int6_dispatch.

These tests validate the eager / trace-time dispatch path — the same code that
torch.export traces through when building the AOTI graph. The Triton kernels
themselves are covered by test_int6_quantized_gemm.py.

The API contract: after importing int6_dispatch, F.linear / nn.Linear with a
CudaDp4aPlanarInt6Tensor weight produce numerically correct results, routed by
batch size (decode M<=4 -> ``triton::int6_quantized_gemm_m{M}``, everything
else -> inline dequant, never an error).

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
    _encode_int8_per_super,
    CudaDp4aPlanarInt6Tensor,
    pack_int6,
    unpack_int6,
)
from executorch.backends.cuda.quantize_op_dispatch.int6_dispatch import _unit_dq_mm_int6


def _require_cuda(tc: unittest.TestCase) -> None:
    if not torch.cuda.is_available():
        tc.skipTest("CUDA required")


def _make_int6_tensor(N, K, group_size=16):
    """Build a CudaDp4aPlanarInt6Tensor (symmetric Q6_K) and return (tensor, q, scale).

    ``q`` (int8 in [-32, 31]) and the returned ``scale`` (the effective per-group
    scale ``code * step`` the kernel actually decodes) are the originals, so tests
    can measure against the exact dequant reference ``w = q * scale``. K must be a
    multiple of 256 (the per-256 scale-step super-block).
    """
    q = torch.randint(-32, 32, (N, K), dtype=torch.int8)
    scale = (torch.rand(N, K // group_size) * 0.1 + 0.01).to(torch.bfloat16)
    ql, qh = pack_int6(q)
    # The tensor stores int8 scale *codes* + a per-256-super-block [N, K/256]
    # fp16 step (scale = code * step[:, g // (256 // gs)]); encode here and
    # return the effective scale so the reference dequant matches the kernel.
    scale_codes, steps = _encode_int8_per_super(scale.float(), group_size)
    n_super = steps.shape[1]
    gps = (K // group_size) // n_super
    step_g = steps.to(torch.bfloat16).repeat_interleave(gps, dim=1)
    eff_scale = (scale_codes.to(torch.bfloat16) * step_g).to(torch.bfloat16)
    t = CudaDp4aPlanarInt6Tensor(
        ql, qh, scale_codes, steps, [1, group_size], torch.Size([N, K])
    )
    return t, q, eff_scale


def _ref_weight(q, scale, group_size, dtype=torch.bfloat16):
    """Exact dequant reference: w[n, k] = q[n, k] * scale[n, k//gs]."""
    N, K = q.shape
    ng = K // group_size
    w = q.to(dtype).reshape(N, ng, group_size) * scale.to(dtype).reshape(N, ng, 1)
    return w.reshape(N, K)


@contextlib.contextmanager
def _record_int6_kernel_ops():
    """Record which INT6 Triton op the dispatch would launch, without a GPU.

    Replaces ``INT6_QUANTIZED_GEMM.op`` with a recorder whose ops compute the
    result via the eager dequant, so the dispatch handler still returns a valid
    tensor.
    """
    from executorch.backends.cuda.triton.kernels.int6_quantized_gemm import (
        INT6_QUANTIZED_GEMM,
    )

    calls = []

    def _op(bucket):
        def run(x, *weight_args):
            calls.append((bucket, tuple(x.shape)))
            return _unit_dq_mm_int6(x, *weight_args)

        return run

    with mock.patch.object(INT6_QUANTIZED_GEMM, "op", side_effect=_op):
        yield calls


class TestDispatchRouting(unittest.TestCase):
    """Type-based routing on CPU: CudaDp4aPlanarInt6Tensor takes inline dequant.

    These tests run without a GPU. Decode on CUDA traces the Triton kernels
    (TestDecodeDispatch); CPU eager cannot launch Triton, so every M takes the
    inline dequant and no INT6 op is reached.
    """

    def setUp(self):
        torch.manual_seed(0)

    def _rel_err(self, out, ref):
        return (
            (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
        ).item()

    def test_cpu_decode_uses_dequant(self):
        """M<=4 on CPU eager takes inline dequant, never a Triton op."""
        t, _, _ = _make_int6_tensor(16, 256)
        x = torch.randn(1, 256, dtype=torch.bfloat16)  # M=1 (decode regime)
        with _record_int6_kernel_ops() as calls:
            out = F.linear(x, t)
        self.assertEqual(calls, [])
        self.assertEqual(out.shape, (1, 16))

    def test_prefill_uses_dequant(self):
        """M>4 uses inline dequant (no custom op) and is numerically correct."""
        t, q, scale = _make_int6_tensor(16, 256)
        x = torch.randn(8, 256, dtype=torch.bfloat16)  # M=8 > 4 (prefill regime)
        with _record_int6_kernel_ops() as calls:
            out = F.linear(x, t)
        self.assertEqual(calls, [])
        ref = F.linear(x, _ref_weight(q, scale, 16))
        self.assertLess(self._rel_err(out, ref), 0.02)

    def test_decode_result_matches_reference(self):
        """The CPU decode-sized result is numerically correct."""
        t, q, scale = _make_int6_tensor(24, 512)
        x = torch.randn(2, 512, dtype=torch.bfloat16)
        with _record_int6_kernel_ops() as calls:
            out = F.linear(x, t)
        self.assertEqual(calls, [])
        ref = F.linear(x, _ref_weight(q, scale, 16))
        self.assertLess(self._rel_err(out, ref), 0.02)

    def test_with_bias(self):
        """Bias is added after the matmul on the decode path."""
        t, q, scale = _make_int6_tensor(16, 256)
        bias = torch.randn(16, dtype=torch.bfloat16)
        x = torch.randn(1, 256, dtype=torch.bfloat16)
        with _record_int6_kernel_ops():
            out = F.linear(x, t, bias)
        ref = F.linear(x, _ref_weight(q, scale, 16), bias)
        self.assertLess(self._rel_err(out, ref), 0.02)

    def test_with_bias_kwarg(self):
        """Bias passed as a keyword (F.linear(x, w, bias=b)) is applied, not dropped."""
        t, q, scale = _make_int6_tensor(16, 256)
        bias = torch.randn(16, dtype=torch.bfloat16)
        x = torch.randn(1, 256, dtype=torch.bfloat16)
        with _record_int6_kernel_ops():
            out = F.linear(x, t, bias=bias)
        ref = F.linear(x, _ref_weight(q, scale, 16), bias)
        self.assertLess(self._rel_err(out, ref), 0.02)
        # Guard against a regression to dropping the keyword bias: the no-bias
        # result must differ from the bias result by exactly the bias.
        with _record_int6_kernel_ops():
            out_no_bias = F.linear(x, t)
        self.assertTrue(
            torch.allclose(out, out_no_bias + bias, atol=1e-2),
            "keyword bias was not applied",
        )

    def test_3d_batched_input(self):
        """3D input is flattened and the output shape is restored."""
        t, q, scale = _make_int6_tensor(16, 256)
        x = torch.randn(2, 8, 256, dtype=torch.bfloat16)  # flattened M=16 > 4
        with _record_int6_kernel_ops() as calls:
            out = F.linear(x, t)
        self.assertEqual(calls, [])  # prefill regime
        self.assertEqual(out.shape, (2, 8, 16))
        ref = F.linear(x, _ref_weight(q, scale, 16))
        self.assertLess(self._rel_err(out, ref), 0.02)

    def test_from_intx_int8_roundtrip(self):
        """_from_intx_int8 packs a symmetric int8 tensor and dispatch is correct."""
        from torchao.quantization import IntxUnpackedToInt8Tensor

        N, K, gs = 16, 256, 16
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
        t = CudaDp4aPlanarInt6Tensor._from_intx_int8(intx)
        x = torch.randn(1, K, dtype=torch.bfloat16)
        with _record_int6_kernel_ops() as calls:
            out = F.linear(x, t)
        self.assertEqual(calls, [])  # CPU eager: inline dequant
        # The packer re-encodes scale as int8 code * per-256 fp16 step, so the
        # reference uses the effective decoded scale (the tensor's dequant), not
        # the raw input scale.
        n_super = t.steps.shape[1]
        gps = (K // gs) // n_super
        eff = (
            t.scale.to(torch.bfloat16)
            * t.steps.to(torch.bfloat16).repeat_interleave(gps, dim=1)
        ).to(torch.bfloat16)
        ref = F.linear(x, _ref_weight(q, eff, gs))
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
            CudaDp4aPlanarInt6Tensor._from_intx_int8(intx)

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
        # bit-pack it losslessly: the unpacked q and the decoded scale match the
        # int8 path. The scale is stored as int8 codes + a per-256 fp16 step
        # (scale = code * step[:, g // (256 // gs)]); the GGUF sub-scales are
        # constant within this single super-block, so the int8 re-encoding is
        # exact here.
        intx = gt.to_intx_unpacked_to_int8_tensor(scale_dtype=torch.bfloat16)
        q_rt = unpack_int6(t.ql, t.qh, N, nb * 256).to(torch.int8)
        self.assertTrue(torch.equal(q_rt, intx.qdata))
        n_groups = intx.scale.shape[1]
        n_super = t.steps.shape[1]
        gps = n_groups // n_super
        step_g = t.steps.to(torch.bfloat16).repeat_interleave(gps, dim=1)
        decoded_scale = (t.scale.to(torch.bfloat16) * step_g).to(torch.bfloat16)
        self.assertTrue(torch.equal(decoded_scale, intx.scale))

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
        t, q, scale = _make_int6_tensor(32, 256)
        ref = _ref_weight(q, scale, 16)
        # Pass an explicit dtype: the scale is stored as int8 codes, so the
        # default (self.scale.dtype) would dequantize in int8 and collapse to 0.
        self.assertTrue(torch.equal(t.dequantize(torch.bfloat16).cpu(), ref))


class TestDecodeDispatch(unittest.TestCase):
    """CUDA export: decode-sized M captures the INT6 bucket op."""

    def setUp(self):
        _require_cuda(self)
        torch.manual_seed(0)

    def _module(self, n=256, k=512, group_size=16):
        t, q, scale = _make_int6_tensor(n, k, group_size)
        module = nn.Linear(k, n, bias=False, dtype=torch.bfloat16)
        module.weight = nn.Parameter(t, requires_grad=False)
        return module.cuda(), _ref_weight(q, scale, group_size).cuda()

    @staticmethod
    def _targets(module, x, dynamic_m=None):
        from torch.export import Dim

        dynamic = None
        if dynamic_m is not None:
            dynamic = ({0: Dim("m", min=dynamic_m[0], max=dynamic_m[1])},)
        with torch.no_grad():
            program = torch.export.export(module, (x,), dynamic_shapes=dynamic)
        return {str(node.target) for node in program.graph.nodes}

    @staticmethod
    def _bucket_ops(targets):
        return {t for t in targets if "int6_quantized_gemm" in t}

    def test_static_m_uses_the_smallest_covering_bucket_op(self):
        module, w_ref = self._module()
        cases = {1: 1, 2: 2, 3: 3, 4: 4, 6: 8, 8: 8, 16: 16, 24: 32, 48: 64, 64: 64}
        for m, bucket in cases.items():
            x = torch.randn(m, 512, dtype=torch.bfloat16, device="cuda")
            ops = self._bucket_ops(self._targets(module, x))
            self.assertEqual(ops, {f"triton.int6_quantized_gemm_m{bucket}.default"}, m)
            with torch.no_grad():
                out = module(x)
            ref = F.linear(x, w_ref)
            rel = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
            self.assertLess(rel.item(), 0.02, m)

    def test_dynamic_m_bounded_by_a_bucket_uses_that_bucket(self):
        module, _ = self._module()
        x = torch.randn(4, 512, dtype=torch.bfloat16, device="cuda")
        self.assertEqual(
            self._bucket_ops(self._targets(module, x, dynamic_m=(2, 4))),
            {"triton.int6_quantized_gemm_m4.default"},
        )
        self.assertEqual(
            self._bucket_ops(self._targets(module, x[:2], dynamic_m=(2, 3))),
            {"triton.int6_quantized_gemm_m3.default"},
        )

    def test_bounded_dynamic_m_through_64_uses_a_bucket(self):
        module, _ = self._module()
        x64 = torch.randn(64, 512, dtype=torch.bfloat16, device="cuda")
        for bounds in ((5, 64), (1, 64)):
            self.assertEqual(
                self._bucket_ops(self._targets(module, x64, dynamic_m=bounds)),
                {"triton.int6_quantized_gemm_m64.default"},
                bounds,
            )

    def test_more_than_64_rows_uses_dequant(self):
        module, _ = self._module()
        x65 = torch.randn(65, 512, dtype=torch.bfloat16, device="cuda")
        self.assertFalse(self._bucket_ops(self._targets(module, x65)))
        self.assertFalse(
            self._bucket_ops(self._targets(module, x65, dynamic_m=(1, 65)))
        )


class TestFallbacks(unittest.TestCase):
    """Inputs the INT6 kernels do not serve take inline dequant: no error, no
    Triton op in the graph, correct output."""

    def setUp(self):
        _require_cuda(self)
        torch.manual_seed(5)

    def _check(self, t, q, scale, group_size, x):
        with _record_int6_kernel_ops() as calls:
            out = F.linear(x, t)
        self.assertEqual(calls, [])
        ref = F.linear(
            x.to(torch.bfloat16), _ref_weight(q, scale, group_size).cuda()
        ).to(out.dtype)
        rel = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
        self.assertLess(rel.item(), 0.05)

    def _cuda_tensor(self, n, k, group_size):
        t, q, scale = _make_int6_tensor(n, k, group_size)
        return t.cuda(), q, scale

    def test_fp16_activation(self):
        t, q, scale = self._cuda_tensor(64, 512, 16)
        self._check(
            t, q, scale, 16, torch.randn(1, 512, dtype=torch.float16, device="cuda")
        )

    def test_non_contiguous_activation(self):
        t, q, scale = self._cuda_tensor(64, 512, 16)
        x = torch.randn(512, 2, dtype=torch.bfloat16, device="cuda").t()
        self._check(t, q, scale, 16, x)

    def test_group_size_other_than_16(self):
        t, q, scale = self._cuda_tensor(64, 512, 32)
        self._check(
            t, q, scale, 32, torch.randn(2, 512, dtype=torch.bfloat16, device="cuda")
        )

    def test_more_than_64_rows(self):
        t, q, scale = self._cuda_tensor(64, 512, 16)
        self._check(
            t, q, scale, 16, torch.randn(65, 512, dtype=torch.bfloat16, device="cuda")
        )

    def test_raw_uint8_scale_codes_are_signed(self):
        """uint8 scale storage holds the same signed codes, on the dequant
        fallback (gs = 32) as in the kernels."""
        t, _, _ = self._cuda_tensor(64, 512, 32)
        codes = t.scale.clone()
        codes[:, ::2] = -codes[:, ::2]
        x = torch.randn(1, 512, dtype=torch.bfloat16, device="cuda")
        ref = _unit_dq_mm_int6(x, t.ql, t.qh, codes, t.steps, 32)
        raw = _unit_dq_mm_int6(x, t.ql, t.qh, codes.view(torch.uint8), t.steps, 32)
        torch.testing.assert_close(raw, ref)
        strided = codes.t().contiguous().t().view(torch.uint8)
        self.assertFalse(strided.is_contiguous())
        torch.testing.assert_close(
            _unit_dq_mm_int6(x, t.ql, t.qh, strided, t.steps, 32), ref
        )


if __name__ == "__main__":
    unittest.main()
