# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for IntxUnpackedToInt8Tensor F.linear dispatch via int8_dispatch.

After importing int8_dispatch, F.linear with a weight-only INT8
IntxUnpackedToInt8Tensor routes decode-sized M to
``triton::int8_quantized_gemm_m{M}`` and everything else (prefill, unbounded
dynamic M, unsupported inputs, CPU eager) to inline dequant + F.linear, never an
error. Other IntxUnpackedToInt8Tensor configurations keep their own dequantize.
The kernels themselves are covered by test_int8_quantized_gemm.py.

    python -m pytest backends/cuda/tests/test_int8_dispatch.py -v
"""

import contextlib
import unittest
from unittest import mock

import executorch.backends.cuda.quantize_op_dispatch.int8_dispatch  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as F
from executorch.backends.cuda.quantize_op_dispatch.int8_dispatch import _unit_dq_mm_int8
from torch.export import Dim
from torchao.quantization.quantize_.workflows.intx.intx_unpacked_to_int8_tensor import (
    IntxUnpackedToInt8Tensor,
)


def _require_cuda(tc: unittest.TestCase) -> None:
    if not torch.cuda.is_available():
        tc.skipTest("CUDA required")


def _make_int8_tensor(n, k, group_size=32, device="cpu", **overrides):
    qdata = torch.randint(-64, 64, (n, k), dtype=torch.int8)
    scale = (torch.rand(n, k // group_size) * 0.02 + 0.005).to(torch.bfloat16)
    zero = torch.randint(-8, 9, (n, k // group_size), dtype=torch.int8)
    kwargs = {
        "qdata": qdata,
        "scale": scale,
        "zero_point": zero,
        "target_dtype": torch.int8,
        "block_size": (1, group_size),
        "dtype": torch.bfloat16,
        "activation_quantization": None,
    }
    kwargs.update(overrides)
    t = IntxUnpackedToInt8Tensor(**kwargs)
    ref = (
        (
            qdata.to(torch.bfloat16).reshape(n, -1, group_size)
            - zero.to(torch.bfloat16)[..., None]
        )
        * scale[..., None]
    ).reshape(n, k)
    return t.to(device), ref.to(device)


@contextlib.contextmanager
def _record_int8_kernel_ops():
    """Record which INT8 Triton op the dispatch would launch; the recorder
    computes the result via the eager dequant."""
    from executorch.backends.cuda.triton.kernels.int8_quantized_gemm import (
        INT8_QUANTIZED_GEMM,
    )

    calls = []

    def _op(bucket):
        def run(x, *weight_args):
            calls.append((bucket, tuple(x.shape)))
            return _unit_dq_mm_int8(x, *weight_args)

        return run

    with mock.patch.object(INT8_QUANTIZED_GEMM, "op", side_effect=_op):
        yield calls


def _rel(out, ref):
    return ((out.float() - ref.float()).abs().mean() / ref.float().abs().mean()).item()


class TestCpuDispatch(unittest.TestCase):
    """CPU eager cannot launch Triton: every M takes inline dequant."""

    def setUp(self):
        torch.manual_seed(0)

    def test_decode_and_prefill_use_dequant(self):
        t, ref_w = _make_int8_tensor(16, 256)
        for m in (1, 4, 8):
            x = torch.randn(m, 256, dtype=torch.bfloat16)
            with _record_int8_kernel_ops() as calls:
                out = F.linear(x, t)
            self.assertEqual(calls, [], m)
            self.assertLess(_rel(out, F.linear(x, ref_w)), 0.01, m)

    def test_bias_positional_and_keyword(self):
        t, ref_w = _make_int8_tensor(16, 256)
        bias = torch.randn(16, dtype=torch.bfloat16)
        x = torch.randn(1, 256, dtype=torch.bfloat16)
        ref = F.linear(x, ref_w, bias)
        self.assertLess(_rel(F.linear(x, t, bias), ref), 0.01)
        self.assertLess(_rel(F.linear(x, t, bias=bias), ref), 0.01)

    def test_3d_input(self):
        t, ref_w = _make_int8_tensor(16, 256)
        x = torch.randn(2, 3, 256, dtype=torch.bfloat16)
        out = F.linear(x, t)
        self.assertEqual(out.shape, (2, 3, 16))
        self.assertLess(_rel(out, F.linear(x, ref_w)), 0.01)

    def test_other_configurations_keep_their_dequantize(self):
        t, _ = _make_int8_tensor(16, 256, target_dtype=torch.int4)
        x = torch.randn(1, 256, dtype=torch.bfloat16)
        with _record_int8_kernel_ops() as calls, mock.patch.object(
            IntxUnpackedToInt8Tensor,
            "dequantize",
            autospec=True,
            side_effect=IntxUnpackedToInt8Tensor.dequantize,
        ) as dequantize:
            F.linear(x, t)
        self.assertEqual(calls, [])
        self.assertEqual(dequantize.call_count, 1)


class _Linear(nn.Module):
    def __init__(self, t):
        super().__init__()
        self.weight = nn.Parameter(t, requires_grad=False)

    def forward(self, x):
        return F.linear(x, self.weight)


class TestDecodeDispatch(unittest.TestCase):
    """CUDA export: decode-sized M captures the INT8 bucket op."""

    def setUp(self):
        _require_cuda(self)
        torch.manual_seed(1)

    @staticmethod
    def _bucket_ops(module, x, dynamic_m=None):
        dynamic = (
            None
            if dynamic_m is None
            else ({0: Dim("m", min=dynamic_m[0], max=dynamic_m[1])},)
        )
        with torch.no_grad():
            program = torch.export.export(module, (x,), dynamic_shapes=dynamic)
        return {
            str(n.target)
            for n in program.graph.nodes
            if "int8_quantized_gemm" in str(n.target)
        }

    def test_decode_sized_m_uses_its_bucket_op(self):
        t, ref_w = _make_int8_tensor(256, 512, device="cuda")
        module = _Linear(t)
        for m in (1, 2, 3, 4):
            x = torch.randn(m, 512, dtype=torch.bfloat16, device="cuda")
            self.assertEqual(
                self._bucket_ops(module, x),
                {f"triton.int8_quantized_gemm_m{m}.default"},
                m,
            )
            with torch.no_grad():
                out = module(x)
            self.assertLess(_rel(out, F.linear(x, ref_w)), 0.01, m)

    def test_dynamic_m_bounded_by_a_bucket_uses_that_bucket(self):
        t, _ = _make_int8_tensor(256, 512, device="cuda")
        module = _Linear(t)
        x = torch.randn(4, 512, dtype=torch.bfloat16, device="cuda")
        self.assertEqual(
            self._bucket_ops(module, x, (2, 4)),
            {"triton.int8_quantized_gemm_m4.default"},
        )
        self.assertEqual(
            self._bucket_ops(module, x[:2], (2, 3)),
            {"triton.int8_quantized_gemm_m3.default"},
        )

    def test_prefill_and_unbounded_dynamic_m_use_dequant(self):
        t, _ = _make_int8_tensor(256, 512, device="cuda")
        module = _Linear(t)
        x8 = torch.randn(8, 512, dtype=torch.bfloat16, device="cuda")
        self.assertFalse(self._bucket_ops(module, x8))
        self.assertFalse(self._bucket_ops(module, x8, (5, 64)))
        self.assertFalse(self._bucket_ops(module, x8, (1, 64)))


class TestFallbacks(unittest.TestCase):
    """Inputs the INT8 kernels do not serve take inline dequant: no error, no
    Triton op, correct output."""

    def setUp(self):
        _require_cuda(self)
        torch.manual_seed(2)

    def _check(self, t, ref_w, x):
        with _record_int8_kernel_ops() as calls:
            out = F.linear(x, t)
        self.assertEqual(calls, [])
        self.assertLess(
            _rel(out, F.linear(x.to(ref_w.dtype), ref_w).to(out.dtype)), 0.01
        )

    def test_fp16_activation(self):
        t, ref_w = _make_int8_tensor(64, 512, device="cuda")
        self._check(t, ref_w, torch.randn(1, 512, dtype=torch.float16, device="cuda"))

    def test_non_contiguous_activation(self):
        t, ref_w = _make_int8_tensor(64, 512, device="cuda")
        self._check(
            t, ref_w, torch.randn(512, 2, dtype=torch.bfloat16, device="cuda").t()
        )

    def test_k_not_a_multiple_of_256(self):
        t, ref_w = _make_int8_tensor(64, 288, device="cuda")
        self._check(t, ref_w, torch.randn(1, 288, dtype=torch.bfloat16, device="cuda"))

    def test_group_size_below_32(self):
        t, ref_w = _make_int8_tensor(64, 512, group_size=16, device="cuda")
        self._check(t, ref_w, torch.randn(2, 512, dtype=torch.bfloat16, device="cuda"))

    def test_more_than_four_rows(self):
        t, ref_w = _make_int8_tensor(64, 512, device="cuda")
        self._check(t, ref_w, torch.randn(5, 512, dtype=torch.bfloat16, device="cuda"))


if __name__ == "__main__":
    unittest.main()
