#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the W4A8 prefill GEMM (triton::int4_w4a8_prefill_gemm).

Two separate checks:

* the kernel against an FP64 evaluation of the exact quantized algorithm
  (same INT8 activations and scales, same FP32-dequantized weights), which
  isolates kernel arithmetic (FP32 epilogue, TF32 zero term, split-K, BF16
  output);
* the quantized result against BF16 activations times the dequantized weight,
  which is the activation-quantization error.

Errors are relative to the output RMS, never per element, so near-zero outputs
do not inflate or hide errors.

Usage:
  python -m pytest backends/cuda/tests/test_int4_w4a8_prefill_gemm.py -v
"""

import os
import unittest
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F
from executorch.backends.cuda.coalesced_int4_tensor import CudaCoalescedInt4Tensor
from executorch.backends.cuda.quantize_op_dispatch import int4_dispatch
from executorch.backends.cuda.triton.kernels import int4_w4a8_prefill_gemm as P


def _require_cuda(tc: unittest.TestCase) -> None:
    if not torch.cuda.is_available():
        tc.skipTest("CUDA required")


def _random_coalesced(N, K, device="cuda", zero_code_max=255):
    """Random CudaCoalescedInt4Tensor storage covering the full code ranges."""
    g = torch.Generator(device="cpu").manual_seed(N * 7 + K)
    qdata = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, generator=g)
    scale = torch.randint(0, 256, (N, K // 32), dtype=torch.uint8, generator=g)
    zero = torch.randint(
        0, zero_code_max + 1, (N, K // 32), dtype=torch.uint8, generator=g
    )
    scale_step = (torch.rand(N, K // 256, generator=g) * 1e-3 + 1e-5).half()
    zero_step = (torch.rand(N, K // 256, generator=g) * 0.07).half()
    return tuple(t.to(device) for t in (qdata, scale, scale_step, zero, zero_step))


def _dequant_fp64(qdata, scale, scale_step, zero, zero_step):
    N, K = qdata.shape[0], qdata.shape[1] * 2
    q = torch.stack([qdata & 0xF, qdata >> 4], dim=-1).reshape(N, K).double()
    s = scale.double() * scale_step.double().repeat_interleave(8, dim=1)
    z = zero.double() * zero_step.double().repeat_interleave(8, dim=1)
    return (q - z.repeat_interleave(32, dim=1)) * s.repeat_interleave(32, dim=1)


def _nrmse(y, ref):
    return ((y.double() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()


def _cos(y, ref):
    return F.cosine_similarity(y.double().flatten(), ref.flatten(), dim=0).item()


class TestConfigSelection(unittest.TestCase):
    def test_class_configs_index_valid_configs(self):
        for N, K in (*P._TABLE, (2048, 1024), (4096, 256)):
            classes = P.class_configs(N, K)
            self.assertEqual([upper for upper, _ in classes], list(P.M_CLASSES))
            for _, cfg in classes:
                self.assertTrue(0 <= cfg < len(P.CONFIGS), (N, K, cfg))
                self.assertLessEqual(P.CONFIGS[cfg].split_k, K // 32)
            self.assertEqual(classes[-1][1], 0)

    def test_narrow_output_keeps_fallback(self):
        self.assertEqual(P.class_configs(1023, 1024), [(None, None)])
        self.assertIsNone(P.config_for_m(300, 1023, 1024))

    def test_config_for_m_class_bounds(self):
        N, K = 6656, 4096
        row = P._TABLE[(N, K)]
        for i, upper in enumerate(P.M_CLASSES[:-1]):
            self.assertEqual(P.CONFIGS[P.config_for_m(upper, N, K)], row[i])
            self.assertEqual(P.CONFIGS[P.config_for_m(upper + 1, N, K)], row[i + 1])


class TestPrefillFlag(unittest.TestCase):
    def setUp(self):
        self.addCleanup(int4_dispatch.set_prefill_w4a8_enabled, None)
        int4_dispatch.set_prefill_w4a8_enabled(None)

    def _enabled_with_env(self, value):
        env = {} if value is None else {int4_dispatch.PREFILL_W4A8_ENV: value}
        with mock.patch.dict(os.environ, env, clear=False):
            if value is None:
                os.environ.pop(int4_dispatch.PREFILL_W4A8_ENV, None)
            return int4_dispatch.prefill_w4a8_enabled()

    def test_environment_read_at_first_use(self):
        self.assertFalse(self._enabled_with_env(None))
        self.assertFalse(self._enabled_with_env("0"))
        self.assertTrue(self._enabled_with_env("1"))

    def test_malformed_value_stays_off_with_warning(self):
        for value in ("abc", "2", " 1", "true", ""):
            with self.assertLogs(int4_dispatch.logger, level="WARNING"):
                int4_dispatch._parse_prefill_w4a8_flag.cache_clear()
                self.assertFalse(self._enabled_with_env(value), value)

    def test_setter_overrides_environment(self):
        int4_dispatch.set_prefill_w4a8_enabled(True)
        self.assertTrue(self._enabled_with_env("0"))
        int4_dispatch.set_prefill_w4a8_enabled(False)
        self.assertFalse(self._enabled_with_env("1"))
        int4_dispatch.set_prefill_w4a8_enabled(None)
        self.assertTrue(self._enabled_with_env("1"))


class TestTargetGuard(unittest.TestCase):
    def _reason(self, hip=None, capability=(8, 0), fake=False, compile_sm=80):
        x = torch.empty(4, 256)
        with (
            mock.patch.object(torch.version, "hip", hip),
            mock.patch.object(
                torch.cuda, "get_device_capability", return_value=capability
            ),
            mock.patch.object(P, "is_fake", return_value=fake),
            mock.patch.object(P, "_compile_target_sm", return_value=compile_sm),
        ):
            return P._target_unsupported_reason(x)

    def test_sm80_and_newer_supported(self):
        for capability in ((8, 0), (9, 0), (12, 0)):
            self.assertIsNone(self._reason(capability=capability), capability)

    def test_pre_sm80_and_rocm_rejected(self):
        self.assertIn("sm_75", self._reason(capability=(7, 5)))
        self.assertIn("ROCm", self._reason(hip="6.2"))
        self.assertIn("ROCm", self._reason(hip="6.2", fake=True))

    def test_compile_target_sm_parsing(self):
        for target, expected in (
            (None, 0),
            ("", 0),
            ("80", 80),
            ("120", 120),
            ("120a", 120),
            ("sm_90", 0),
        ):
            with mock.patch(
                "torch._inductor.codegen.cuda.compile_utils._nvcc_arch_as_compile_option",
                return_value=target,
            ):
                self.assertEqual(P._compile_target_sm(), expected, target)

    def test_fake_tensors_check_compile_target(self):
        self.assertIsNone(self._reason(capability=(7, 5), fake=True, compile_sm=90))
        self.assertIn("sm_75", self._reason(fake=True, compile_sm=75))


class TestActivationQuantization(unittest.TestCase):
    def setUp(self):
        _require_cuda(self)

    def _check(self, x):
        xq, xscale, xga = P.quantize_activations_w4a8_prefill(x)
        rq, rs, rg = P.reference_quantize_activations(x)
        self.assertTrue(torch.equal(xq, rq))
        torch.testing.assert_close(xscale, rs, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(xga, rg.float(), rtol=0, atol=0, equal_nan=True)
        return xq, xscale, xga

    def test_random_and_tails(self):
        for M in (1, 17, 65, 300):
            self._check(torch.randn(M, 512, device="cuda").bfloat16())

    def test_adversarial(self):
        K = 768
        x = torch.randn(8, K, device="cuda")
        x[0] = 0.0  # all-zero row
        x[1, 5] = 3.0e4  # single extreme value
        x[2] = 1.5  # constant row (ties at +-127)
        x[3] = torch.randn(K, device="cuda") * 1e-30  # tiny values
        x[4, ::2] = -6.0e4  # bf16-range extremes, mixed signs
        x[4, 1::2] = 6.0e4
        x[5, :32] = 0.0  # one all-zero group
        self._check(x.bfloat16())

    def test_non_finite_group_scale_is_nan(self):
        x = torch.randn(4, 512, device="cuda")
        x[0, 40] = float("nan")
        x[1, 3] = float("inf")
        x[2, 300] = float("-inf")
        xq, xscale, xga = self._check(x.bfloat16())
        for row, group, col in ((0, 1, 40), (1, 0, 3), (2, 9, 300)):
            self.assertTrue(xscale[row, group].isnan().item(), row)
            self.assertTrue(xga[row, group].isnan().item(), row)
            self.assertEqual(xq[row, col].item(), 0, row)
            self.assertEqual(int(xscale[row].isnan().sum()), 1, row)
        self.assertTrue(torch.isfinite(xscale[3]).all())

    def test_tiny_group_where_inverse_scale_overflows(self):
        # 127 / amax overflows FP32 for amax below ~3.7e-37.
        x = torch.zeros(3, 256, device="cuda")
        x[0, :32] = torch.linspace(-2e-38, 2e-38, 32, device="cuda")
        x[1, 32:64] = torch.randn(32, device="cuda") * 1e-38
        x[1, 40] = 0.0
        x[2, :8] = 9e-40  # BF16 subnormals
        x = x.bfloat16()
        xq, xscale, _ = self._check(x)
        self.assertTrue(torch.isfinite(xscale).all())
        self.assertTrue(torch.equal(xq[x == 0], torch.zeros_like(xq[x == 0])))
        self.assertEqual(int(xq[0, :32].abs().max()), 127)
        self.assertEqual(int(xq[1, 32:64].abs().max()), 127)
        self.assertTrue(torch.equal(xq[2, :8], torch.full_like(xq[2, :8], 127)))


class TestPrefillGemm(unittest.TestCase):
    def setUp(self):
        _require_cuda(self)
        torch.manual_seed(0)

    def test_matches_exact_algorithm(self):
        for M, N, K in (
            (5, 200, 512),
            (65, 200, 512),
            (130, 1000, 1024),
            (257, 4100, 768),
            (512, 256, 6656),
        ):
            w = _random_coalesced(N, K)
            x = torch.randn(M, K, device="cuda").bfloat16()
            ref = P.reference_w4a8_prefill(x, *w)
            for c in sorted({P.config_for_m(M, N, K) or 0, 0}):
                y = P.int4_w4a8_prefill_gemm(x, *w, config=c)
                self.assertEqual(y.shape, (M, N))
                self.assertEqual(y.dtype, torch.bfloat16)
                self.assertLess(_nrmse(y, ref), 4e-3, (M, N, K, c))

    def test_every_table_config_matches(self):
        M, N, K = 200, 1300, 1024
        w = _random_coalesced(N, K)
        x = torch.randn(M, K, device="cuda").bfloat16()
        ref = P.reference_w4a8_prefill(x, *w)
        for c in range(len(P.CONFIGS)):
            y = P.int4_w4a8_prefill_gemm(x, *w, config=c)
            self.assertLess(_nrmse(y, ref), 4e-3, P.CONFIGS[c])

    def test_zero_modes_and_uneven_split(self):
        # K/32 = 40 groups: splits of 3 and 6 leave uneven ranges; zero modes agree.
        M, N, K = 70, 384, 1280
        w = _random_coalesced(N, K)
        x = torch.randn(M, K, device="cuda").bfloat16()
        ref = P.reference_w4a8_prefill(x, *w)
        for sk in (1, 3, 6):
            for zm in (P.ZERO_TAIL_TF32, P.ZERO_TAIL_TF32X3, P.ZERO_EPI_CODES):
                cfg = P.LaunchConfig(64, 128, 4, 3, sk, zm)
                self.assertLess(
                    _nrmse(P.w4a8_prefill_gemm(x, *w, cfg=cfg), ref), 4e-3, cfg
                )

    def test_split_k_is_deterministic(self):
        M, N, K = 33, 640, 2048
        w = _random_coalesced(N, K)
        x = torch.randn(M, K, device="cuda").bfloat16()
        cfg = P.LaunchConfig(16, 128, 4, 3, 8, P.ZERO_TAIL_TF32)
        first = P.w4a8_prefill_gemm(x, *w, cfg=cfg)
        for _ in range(3):
            self.assertTrue(torch.equal(first, P.w4a8_prefill_gemm(x, *w, cfg=cfg)))

    def test_quantization_error_vs_bf16_activation(self):
        M, N, K = 256, 2048, 4096
        w = _random_coalesced(N, K, zero_code_max=128)
        x = torch.randn(M, K, device="cuda").bfloat16()
        ref = x.double() @ _dequant_fp64(*w).t()
        y = P.int4_w4a8_prefill_gemm(x, *w)
        self.assertLess(_nrmse(y, ref), 1e-2)
        self.assertGreater(_cos(y, ref), 0.9999)

    def test_zero_activation_gives_zero(self):
        w = _random_coalesced(512, 1024)
        for M, c in ((100, 0), (7, P.config_for_m(7, 512, 1024) or 0)):
            x = torch.zeros(M, 1024, device="cuda", dtype=torch.bfloat16)
            self.assertEqual(
                P.int4_w4a8_prefill_gemm(x, *w, config=c).abs().max().item(), 0.0
            )

    def test_extreme_activation_rows(self):
        M, N, K = 96, 640, 1024
        w = _random_coalesced(N, K)
        x = torch.randn(M, K, device="cuda")
        x[0, 7] = 4.0e4
        x[1] = 0.0
        x[2] *= 1e-20
        x = x.bfloat16()
        y = P.int4_w4a8_prefill_gemm(x, *w)
        ref = P.reference_w4a8_prefill(x, *w)
        self.assertTrue(torch.isfinite(y).all())
        self.assertEqual(y[1].abs().max().item(), 0.0)
        for r in (0, 2, 5):
            self.assertLess(_nrmse(y[r : r + 1], ref[r : r + 1]), 4e-3, r)

    def test_non_finite_activation_rows_are_nan(self):
        M, N, K = 6, 1024, 512
        w = _random_coalesced(N, K)
        x = torch.randn(M, K, device="cuda")
        x[1, 5] = float("nan")
        x[3, 100] = float("inf")
        x[4, 511] = float("-inf")
        x = x.bfloat16()
        finite_rows = [0, 2, 5]
        ref = P.reference_w4a8_prefill(x[finite_rows], *w)
        for c in sorted({P.config_for_m(M, N, K), 0}):
            y = P.int4_w4a8_prefill_gemm(x, *w, config=c)
            for r in (1, 3, 4):
                self.assertTrue(y[r].isnan().all().item(), (c, r))
            self.assertLess(_nrmse(y[finite_rows], ref), 4e-3, c)

    def test_tiny_activation_rows_stay_finite(self):
        M, N, K = 4, 1024, 512
        w = _random_coalesced(N, K)
        x = torch.randn(M, K, device="cuda") * 1e-38
        x[0] = 0.0
        x[2, :64] = 0.0
        x = x.bfloat16()
        for c in sorted({P.config_for_m(M, N, K), 0}):
            y = P.int4_w4a8_prefill_gemm(x, *w, config=c)
            self.assertTrue(torch.isfinite(y).all().item(), c)
            self.assertEqual(y[0].abs().max().item(), 0.0, c)

    def test_non_contiguous_activation(self):
        w = _random_coalesced(300, 512)
        x = torch.randn(130, 1024, device="cuda").bfloat16()[:, :512]
        self.assertFalse(x.is_contiguous())
        y = P.int4_w4a8_prefill_gemm(x, *w)
        ref = P.reference_w4a8_prefill(x.contiguous(), *w)
        self.assertLess(_nrmse(y, ref), 4e-3)

    def test_unsupported_inputs_rejected(self):
        w = _random_coalesced(64, 512)
        x = torch.randn(80, 512, device="cuda").bfloat16()
        self.assertIsNone(P.prefill_unsupported_reason(x, *w, 32))
        self.assertIsNotNone(P.prefill_unsupported_reason(x, *w, 64))
        self.assertIsNotNone(P.prefill_unsupported_reason(x.half(), *w, 32))
        x_bad_k = torch.randn(80, 320, device="cuda").bfloat16()
        self.assertIsNotNone(P.prefill_unsupported_reason(x_bad_k, *w, 32))
        with self.assertRaises(RuntimeError):
            P.int4_w4a8_prefill_gemm(x, *w, config=len(P.CONFIGS))

    def test_malformed_metadata_and_devices_never_launch(self):
        qdata, scale, scale_step, zero, zero_step = _random_coalesced(64, 512)
        x = torch.randn(80, 512, device="cuda").bfloat16()
        bad = {
            "short_scale": (
                qdata,
                scale[:, :-1].contiguous(),
                scale_step,
                zero,
                zero_step,
            ),
            "short_zero_step": (
                qdata,
                scale,
                scale_step,
                zero,
                zero_step[:, :1].contiguous(),
            ),
            "rows_mismatch": (
                qdata,
                scale[:-1].contiguous(),
                scale_step,
                zero,
                zero_step,
            ),
            "fp32_step": (qdata, scale, scale_step.float(), zero, zero_step),
            "int8_qdata": (qdata.view(torch.int8), scale, scale_step, zero, zero_step),
            "strided_zero": (
                qdata,
                scale,
                scale_step,
                torch.cat([zero, zero], 1)[:, ::2],
                zero_step,
            ),
            "weights_on_cpu": (
                qdata.cpu(),
                scale.cpu(),
                scale_step.cpu(),
                zero.cpu(),
                zero_step.cpu(),
            ),
            "empty_n": tuple(
                t[:0] for t in (qdata, scale, scale_step, zero, zero_step)
            ),
        }
        for name, w in bad.items():
            self.assertIsNotNone(P.prefill_unsupported_reason(x, *w, 32), name)
            with self.assertRaises(RuntimeError, msg=name):
                P.int4_w4a8_prefill_gemm(x, *w)
        self.assertIsNotNone(
            P.prefill_unsupported_reason(
                x.cpu(), *(t.cpu() for t in bad["short_scale"]), 32
            )
        )

    def test_empty_m(self):
        w = _random_coalesced(64, 512)
        y = P.int4_w4a8_prefill_gemm(
            torch.empty(0, 512, device="cuda", dtype=torch.bfloat16), *w
        )
        self.assertEqual(tuple(y.shape), (0, 64))

    def test_output_offsets_beyond_int32(self):
        free, _ = torch.cuda.mem_get_info()
        M, N, K = 8256, 262144, 256  # M * N > 2**31 output elements
        if free < 6 * 2**30:
            self.skipTest("needs ~6 GiB free GPU memory")
        w = _random_coalesced(N, K)
        x = torch.randn(M, K, device="cuda").bfloat16()
        y = P.int4_w4a8_prefill_gemm(x, *w)
        rows = torch.tensor([0, M // 2, M - 1], device="cuda")
        cols = slice(N - 4096, N)
        ref = P.reference_w4a8_prefill(x[rows], *(t[cols] for t in w))
        self.assertLess(_nrmse(y[rows, cols], ref), 4e-3)
        del y

    def test_split_partials_beyond_int32(self):
        free, _ = torch.cuda.mem_get_info()
        M, N, K = 64, 4_300_000, 256  # split_k * M * N > 2**31 partial elements
        cfg = P.LaunchConfig(64, 128, 4, 3, 8, P.ZERO_TAIL_TF32)
        if free < 12 * 2**30:
            self.skipTest("needs ~12 GiB free GPU memory")
        w = _random_coalesced(N, K)
        x = torch.randn(M, K, device="cuda").bfloat16()
        y = P.w4a8_prefill_gemm(x, *w, cfg=cfg)
        cols = slice(N - 2048, N)
        ref = P.reference_w4a8_prefill(x, *(t[cols] for t in w))
        self.assertLess(_nrmse(y[:, cols], ref), 4e-3)


class TestDispatch(unittest.TestCase):
    def setUp(self):
        _require_cuda(self)
        torch.manual_seed(0)
        self.addCleanup(int4_dispatch.set_prefill_w4a8_enabled, None)
        self.linear, self.weight = self._linear(2048, 1024)

    @staticmethod
    def _linear(N, K):
        qdata, scale, scale_step, zero, zero_step = _random_coalesced(N, K)
        weight = CudaCoalescedInt4Tensor(
            qdata, scale, scale_step, zero, zero_step, [1, 32], torch.Size([N, K])
        )
        linear = nn.Linear(K, N, bias=False, device="cuda", dtype=torch.bfloat16)
        linear.weight = nn.Parameter(weight, requires_grad=False)
        return linear, weight

    def _call(self, M, enabled, linear=None):
        linear = linear or self.linear
        int4_dispatch.set_prefill_w4a8_enabled(enabled)
        x = torch.randn(M, linear.in_features, device="cuda").bfloat16()
        with (
            mock.patch.object(
                P, "int4_w4a8_prefill_gemm", wraps=P.int4_w4a8_prefill_gemm
            ) as op,
            mock.patch.object(
                int4_dispatch, "_dequant_matmul", wraps=int4_dispatch._dequant_matmul
            ) as deq,
        ):
            y = linear(x)
        return x, y, op.call_count, deq.call_count

    def test_prefill_routes_to_w4a8_when_enabled(self):
        x, y, op_calls, deq_calls = self._call(300, True)
        self.assertEqual((op_calls, deq_calls), (1, 0))
        ref = x.double() @ self.weight.dequantize(torch.float32).double().t()
        self.assertGreater(_cos(y, ref), 0.9999)

    def test_prefill_uses_dequant_when_disabled(self):
        _, _, op_calls, deq_calls = self._call(300, False)
        self.assertEqual((op_calls, deq_calls), (0, 1))

    def test_narrow_weight_keeps_dequant(self):
        narrow, _ = self._linear(256, 1024)
        self.assertEqual(self._call(300, True, narrow)[2:], (0, 1))

    def test_decode_buckets_unchanged(self):
        for M in (1, 4, 64):
            _, _, op_calls, deq_calls = self._call(M, True)
            self.assertEqual((op_calls, deq_calls), (0, 0), M)

    def test_3d_activation(self):
        int4_dispatch.set_prefill_w4a8_enabled(True)
        x = torch.randn(2, 150, 1024, device="cuda").bfloat16()
        y = self.linear(x)
        ref = self.linear(x.reshape(300, 1024)).reshape(2, 150, -1)
        self.assertEqual(y.shape, (2, 150, 2048))
        self.assertTrue(torch.equal(y, ref))

    def test_export_dynamic_m_uses_large_m_config(self):
        int4_dispatch.set_prefill_w4a8_enabled(True)
        x = torch.randn(300, 1024, device="cuda").bfloat16()
        m = torch.export.Dim("m", min=5, max=4096)
        ep = torch.export.export(
            self.linear, (x,), dynamic_shapes=({0: m},), strict=False
        )
        nodes = [
            n for _, mod in ep.graph_module.named_modules() for n in mod.graph.nodes
        ]
        prefill = [
            n for n in nodes if str(n.target) == "triton.int4_w4a8_prefill_gemm.default"
        ]
        self.assertEqual(len(prefill), 1)
        config = prefill[0].kwargs.get(
            "config", prefill[0].args[6] if len(prefill[0].args) > 6 else 0
        )
        self.assertEqual(config, 0)
        self.assertNotIn("cond", {str(n.target) for n in nodes})
        module = ep.module()
        weight_args = tuple(
            getattr(self.weight, n)
            for n in ("qdata", "scale", "scale_step", "zero_point", "zero_point_step")
        )
        for M in (5, 64, 65, 300, 1500):
            xm = torch.randn(M, 1024, device="cuda").bfloat16()
            eager = P.int4_w4a8_prefill_gemm(xm, *weight_args, config=0)
            self.assertTrue(torch.equal(module(xm), eager), M)

    def test_static_m_selects_class_config(self):
        int4_dispatch.set_prefill_w4a8_enabled(True)
        for M in (65, 300, 1500):
            x = torch.randn(M, 1024, device="cuda").bfloat16()
            with mock.patch.object(
                P, "int4_w4a8_prefill_gemm", wraps=P.int4_w4a8_prefill_gemm
            ) as op:
                self.linear(x)
            self.assertEqual(
                op.call_args.kwargs["config"], P.config_for_m(M, 2048, 1024), M
            )
