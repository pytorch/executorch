#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Int4Tensor F.linear dispatch via quantize_op_dispatch.int4_dispatch.

These tests validate the eager / trace-time dispatch path — the same code
that torch.export traces through when building the AOTI graph. The decode
Triton kernels themselves are covered by test_int4_quantized_gemm.py.

The API contract: after importing int4_dispatch, F.linear and nn.Linear
with Int4Tensor weights produce numerically correct results. Tests verify
this across the Triton path (M<=64), fallback (M>64), batched (3D), bias, group sizes,
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
from executorch.examples.models.gemma4_31b.cuda_packers import pack_linear_for_cuda
from executorch.extension.llm.export.int4 import ExportableInt4Tensor
from executorch.extension.llm.export.quant.quantize import (
    dequantize_weight,
    quantize_weight,
)
from executorch.extension.llm.export.quant.recipe import QuantConfig


def _require_cuda(tc: unittest.TestCase) -> None:
    if not torch.cuda.is_available():
        tc.skipTest("CUDA required")


def _make_int4_linear(N, K, group_size=128, symmetric=False, bias=False):
    """Build an nn.Linear with ExportableInt4Tensor weight + bf16 ref weight.

    Mirrors production: weights are converted to ExportableInt4Tensor (the
    canonical portable int4 form) before packing for CUDA. The bf16 reference
    is the original unquantized weight, so tests can measure quantization
    error against the true value.
    """
    w_bf16 = torch.randn(N, K, dtype=torch.bfloat16)
    config = QuantConfig(
        bits=4, group_size=group_size, symmetric=symmetric, method="min_max"
    )
    exportable_w = ExportableInt4Tensor.from_int4_tensor(
        quantize_weight(w_bf16, config)
    )

    # device="cuda" so the random init draws from the CUDA RNG to match the
    # same random weight as regular int4 dispatch and fit the same numerical
    # error tolerance.
    module = nn.Linear(K, N, bias=bias, dtype=torch.bfloat16, device="cuda")
    pack_linear_for_cuda(module, {"weight": exportable_w})
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
        # The same kernel either way; outputs reach ~85, where a bf16 ulp is
        # 0.5, so compare relative to the output, not absolutely.
        self._check(out_compiled, out_eager, tol=0.01)


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
        exportable_w = ExportableInt4Tensor.from_int4_tensor(
            quantize_weight(w_bf16, config)
        )
        module = nn.Linear(512, 256, bias=False)
        pack_linear_for_cuda(module, {"weight": exportable_w})
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


def _make_exportable_int4_tensor(N, K, group_size=128, symmetric=False):
    """Build an ``ExportableInt4Tensor`` (canonical portable int4) + bf16 ref."""
    t, w = _make_int4_tensor(N, K, group_size=group_size, symmetric=symmetric)
    return ExportableInt4Tensor.from_int4_tensor(t), w


@contextlib.contextmanager
def _record_int4_kernel_ops():
    """Record which INT4 Triton op the dispatch would launch, without a GPU.

    Replaces ``INT4_QUANTIZED_GEMM.op`` with a recorder whose ops compute the
    result via the eager dequant, so the dispatch handler still returns a valid
    tensor.
    """
    from executorch.backends.cuda.triton.kernels.int4_quantized_gemm import (
        INT4_QUANTIZED_GEMM,
    )

    calls = []

    def _op(bucket):
        def run(x, *weight_args):
            calls.append((bucket, tuple(x.shape)))
            return _dequant_matmul(x, *weight_args)

        return run

    with mock.patch.object(INT4_QUANTIZED_GEMM, "op", side_effect=_op):
        yield calls


class TestDispatchRouting(unittest.TestCase):
    """Type-based routing on CPU: CudaCoalescedInt4Tensor takes inline dequant.

    These tests run without a GPU. Decode on CUDA traces the Triton kernels
    (TestDecodeDispatch); on CPU every M takes the inline dequant, and nothing
    reaches the INT4 Triton ops.
    The CUDA path must be selected by weight *type*, not by globally overriding
    torchao ``Int4Tensor``'s F.linear.
    """

    def setUp(self):
        torch.manual_seed(0)

    def _rel_err(self, out, ref):
        return (
            (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
        ).item()

    def test_stock_int4tensor_does_not_route_to_the_int4_ops(self):
        """A plain torchao Int4Tensor must fall back to torchao's default path."""
        t, _ = _make_int4_tensor(16, 64, group_size=32)
        x = torch.randn(1, 64, dtype=torch.bfloat16)  # M=1 (decode regime)
        with _record_int4_kernel_ops() as calls:
            # torchao's default path uses mslk/CUDA and is not exercised on CPU;
            # we only assert that our decode op is NOT reached.
            with contextlib.suppress(Exception):
                F.linear(x, t)
        self.assertEqual(calls, [])

    def test_coalesced_tensor_decode_on_cpu_uses_dequant(self):
        """M<=4 on CPU takes inline dequant, never a Triton op."""
        t, _ = _make_exportable_int4_tensor(16, 256, group_size=32)
        c = CudaCoalescedInt4Tensor.from_exportable_int4_tensor(t)
        x = torch.randn(1, 256, dtype=torch.bfloat16)  # M=1 (decode regime)
        with _record_int4_kernel_ops() as calls:
            out = F.linear(x, c)
        self.assertEqual(calls, [])
        self.assertEqual(out.shape, (1, 16))
        # The coalesced scale/zero are re-encoded as uint8 codes, and one row
        # averages little of that error away.
        ref = F.linear(x, dequantize_weight(t, torch.bfloat16))
        self.assertLess(self._rel_err(out, ref), 0.03)

    def test_coalesced_tensor_prefill_uses_dequant(self):
        """M>4 uses inline dequant (no custom op) and is numerically correct."""
        t, _ = _make_exportable_int4_tensor(16, 256, group_size=32)
        c = CudaCoalescedInt4Tensor.from_exportable_int4_tensor(t)
        x = torch.randn(8, 256, dtype=torch.bfloat16)  # M=8 > 4 (prefill regime)
        with _record_int4_kernel_ops() as calls:
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
        with _record_int4_kernel_ops() as calls:
            with contextlib.suppress(Exception):
                F.linear(x, t)
        self.assertEqual(calls, [])

    def test_from_exportable_int4_tensor_transpose_correct(self):
        """from_exportable_int4_tensor owns the (n_groups, N) -> (N, n_groups) transpose."""
        t, _ = _make_exportable_int4_tensor(24, 256, group_size=64)
        c = CudaCoalescedInt4Tensor.from_exportable_int4_tensor(t)
        n_groups = 256 // 64
        self.assertEqual(tuple(t.scale.shape), (n_groups, 24))  # torchao layout
        self.assertEqual(tuple(c.scale.shape), (24, n_groups))  # coalesced layout
        # Scale is a uint8 code + a per-256 fp16 step; zero is a uint8 code + a
        # per-256 fp16 step. Decoding must recover the transposed torchao
        # scale/zero (within code quant error).
        n_super = int(c.scale_step.shape[1])
        gps = n_groups // n_super
        scale_step_g = c.scale_step.to(torch.bfloat16).repeat_interleave(gps, dim=1)
        dec_scale = c.scale.to(torch.bfloat16) * scale_step_g
        zero_point_step_g = c.zero_point_step.to(torch.bfloat16).repeat_interleave(
            gps, dim=1
        )
        dec_zero = c.zero_point.to(torch.bfloat16) * zero_point_step_g
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
        with _record_int4_kernel_ops() as calls:
            out = F.linear(x, c)
        self.assertEqual(calls, [])
        ref = F.linear(x, dequantize_weight(t, torch.bfloat16))
        self.assertLess(self._rel_err(out, ref), 0.02)


class TestDecodeDispatch(unittest.TestCase):
    """M <= 64 traces the smallest covering Triton bucket."""

    def setUp(self):
        _require_cuda(self)
        torch.manual_seed(42)

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
        return {t for t in targets if "int4_quantized_gemm" in t}

    def test_static_m_uses_the_smallest_covering_bucket_op(self):
        module, w_ref = _make_int4_linear(256, 512, group_size=32)
        cases = {
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            6: 8,
            8: 8,
            16: 16,
            24: 32,
            32: 32,
            48: 64,
            64: 64,
        }
        for m, bucket in cases.items():
            x = torch.randn(m, 512, dtype=torch.bfloat16, device="cuda")
            targets = self._targets(module, x)
            ops = self._bucket_ops(targets)
            self.assertEqual(len(ops), 1, (m, targets))
            self.assertIn(f"int4_quantized_gemm_m{bucket}", next(iter(ops)))
            self.assertFalse(any("constant_pad" in t for t in targets), m)
            with torch.no_grad():
                out = module(x)
            ref = F.linear(x, w_ref)
            self.assertEqual(out.shape, ref.shape)
            rel = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
            self.assertLess(rel.item(), 0.15, m)

    def test_export_from_cpu_inputs_uses_a_bucket_op(self):
        """Export traces with fake tensors, so CPU example inputs and weights
        still capture the Triton kernel."""
        module, _ = _make_int4_linear(256, 512, group_size=32)
        module = module.cpu()
        x = torch.randn(1, 512, dtype=torch.bfloat16)
        targets = self._targets(module, x)
        ops = self._bucket_ops(targets)
        self.assertEqual(len(ops), 1, targets)
        self.assertIn("int4_quantized_gemm_m1", next(iter(ops)))

    def test_dynamic_m_bounded_by_a_bucket_uses_that_bucket(self):
        """A dynamic M in [2, 4] (e.g. a speculative block) takes the 4-row
        bucket, as it took the shim before, not inline dequant."""
        module, _ = _make_int4_linear(256, 512, group_size=32)
        x = torch.randn(4, 512, dtype=torch.bfloat16, device="cuda")
        targets = self._targets(module, x, dynamic_m=(2, 4))
        ops = self._bucket_ops(targets)
        self.assertEqual(len(ops), 1, targets)
        self.assertIn("int4_quantized_gemm_m4", next(iter(ops)))
        targets = self._targets(module, x[:2], dynamic_m=(2, 3))
        self.assertIn("int4_quantized_gemm_m3", next(iter(self._bucket_ops(targets))))

    def test_bounded_dynamic_m_through_64_uses_a_bucket(self):
        module, _ = _make_int4_linear(256, 512, group_size=32)
        x64 = torch.randn(64, 512, dtype=torch.bfloat16, device="cuda")
        for bounds in ((5, 64), (1, 64)):
            targets = self._targets(module, x64, dynamic_m=bounds)
            self.assertIn(
                "int4_quantized_gemm_m64",
                next(iter(self._bucket_ops(targets))),
            )

    def test_more_than_64_rows_uses_dequant(self):
        module, _ = _make_int4_linear(256, 512, group_size=32)
        x65 = torch.randn(65, 512, dtype=torch.bfloat16, device="cuda")
        self.assertFalse(self._bucket_ops(self._targets(module, x65)))
        self.assertFalse(
            self._bucket_ops(self._targets(module, x65, dynamic_m=(1, 65)))
        )

    def test_other_group_sizes_use_dequant(self):
        # With group size 32 the packed tensor already requires K % 256 == 0,
        # so group size is the only weight property that opts out.
        module, _ = _make_int4_linear(256, 512, group_size=128)
        x = torch.randn(1, 512, dtype=torch.bfloat16, device="cuda")
        targets = self._targets(module, x)
        self.assertFalse(self._bucket_ops(targets))


class TestFallbacks(unittest.TestCase):
    """Inputs the INT4 kernels do not serve take inline dequant: no error, no
    Triton op in the graph, correct output. (K not a multiple of 256 cannot
    reach dispatch: CudaCoalescedInt4Tensor packing rejects it; the kernel-side
    rule is covered by test_int4_quantized_gemm.)"""

    def setUp(self):
        _require_cuda(self)
        torch.manual_seed(7)

    def _check(self, module, w_ref, x):
        with _record_int4_kernel_ops() as calls:
            with torch.no_grad():
                out = module(x)
        self.assertEqual(calls, [])
        ref = F.linear(x.to(w_ref.dtype), w_ref).to(out.dtype)
        rel = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
        # Against the unquantized weight, so this is the INT4 quantization error.
        self.assertLess(rel.item(), 0.15)
        with torch.no_grad():
            program = torch.export.export(module, (x,))
        targets = {str(node.target) for node in program.graph.nodes}
        self.assertFalse(any("int4_quantized_gemm" in t for t in targets), targets)

    def test_fp16_activation(self):
        module, w_ref = _make_int4_linear(256, 512, group_size=32)
        self._check(
            module, w_ref, torch.randn(1, 512, dtype=torch.float16, device="cuda")
        )

    def test_non_contiguous_activation(self):
        module, w_ref = _make_int4_linear(256, 512, group_size=32)
        x = torch.randn(512, 2, dtype=torch.bfloat16, device="cuda").t()
        self.assertFalse(x.is_contiguous())
        self._check(module, w_ref, x)

    def test_group_size_other_than_32(self):
        module, w_ref = _make_int4_linear(256, 512, group_size=64)
        self._check(
            module, w_ref, torch.randn(2, 512, dtype=torch.bfloat16, device="cuda")
        )

    def test_more_than_64_rows(self):
        module, w_ref = _make_int4_linear(256, 512, group_size=32)
        self._check(
            module, w_ref, torch.randn(65, 512, dtype=torch.bfloat16, device="cuda")
        )


if __name__ == "__main__":
    unittest.main()
