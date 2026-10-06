# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for quantized F.linear dispatch helpers (quantize_op_dispatch/_gemm_family_dispatch.py).

    python -m pytest backends/cuda/tests/test_gemm_family_dispatch.py -v
"""

import unittest
from unittest import mock

import torch
import torch.nn.functional as F
from executorch.backends.cuda.quantize_op_dispatch import _gemm_family_dispatch as gd
from executorch.backends.cuda.quantize_op_dispatch._gemm_family_dispatch import (
    can_launch_triton,
    chunked_dequant_linear,
    quantized_linear,
    select_bucket,
)
from executorch.backends.cuda.triton.kernels.quantized_gemm_family import (
    QuantizedGemmFamily,
)
from torch.export import Dim
from torch.fx.experimental.symbolic_shapes import statically_known_true


def _prototype(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def _launch(bucket: int, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x @ weight.t() + bucket


def _fake(bucket: int, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]))


def _rows_fit(bucket: int, x: torch.Tensor, weight: torch.Tensor):
    m = x.shape[0]
    if isinstance(m, int):
        return None if m == bucket else "static M must equal the bucket"
    if statically_known_true(m >= 1) and statically_known_true(m <= bucket):
        return None
    return "dynamic M is not provably within the bucket"


_TOY = QuantizedGemmFamily(
    "gemm_family_dispatch_test_toy", (1, 2, 4), _prototype, _launch, _fake, _rows_fit
)


def _dequant(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(x, weight)


class _Routed(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 16))

    def forward(self, x):
        return quantized_linear(
            _TOY, x, (self.weight,), None, lambda x2d: _dequant(x2d, self.weight)
        )


def _toy_ops(program) -> set[str]:
    return {
        str(node.target)
        for node in program.graph.nodes
        if node.op == "call_function" and "gemm_family_dispatch_test_toy" in str(node.target)
    }


class SelectBucketTest(unittest.TestCase):
    def test_static_m_takes_its_own_bucket(self) -> None:
        w = torch.randn(8, 16)
        self.assertEqual(select_bucket(_TOY, torch.randn(1, 16), w), 1)
        self.assertEqual(select_bucket(_TOY, torch.randn(4, 16), w), 4)
        self.assertIsNone(select_bucket(_TOY, torch.randn(3, 16), w))
        self.assertIsNone(select_bucket(_TOY, torch.randn(8, 16), w))

    def test_first_supporting_bucket_in_ascending_order(self) -> None:
        w, x = torch.randn(8, 16), torch.randn(2, 16)
        with mock.patch.object(_TOY, "supports", side_effect=lambda b, *a: b >= 2) as supports:
            self.assertEqual(select_bucket(_TOY, x, w), 2)
        self.assertEqual([c.args[0] for c in supports.call_args_list], [1, 2])
        with mock.patch.object(_TOY, "supports", return_value=False):
            self.assertIsNone(select_bucket(_TOY, x, w))

    def test_exported_dynamic_m_takes_the_smallest_covering_bucket(self) -> None:
        cases = {(1, 2): "m2", (2, 3): "m4", (1, 4): "m4", (2, 5): None}
        for (low, high), expected in cases.items():
            x = torch.randn(high, 16)
            program = torch.export.export(
                _Routed(), (x,), dynamic_shapes=({0: Dim("m", min=low, max=high)},)
            )
            ops = _toy_ops(program)
            if expected is None:
                self.assertEqual(ops, set(), (low, high))
            else:
                self.assertEqual(
                    ops, {f"triton.gemm_family_dispatch_test_toy_{expected}.default"}, (low, high)
                )


class QuantizedLinearTest(unittest.TestCase):
    def test_cpu_eager_falls_back(self) -> None:
        x = torch.randn(1, 16)
        self.assertFalse(can_launch_triton(x))
        out = _Routed()(x)
        self.assertEqual(out.shape, (1, 8))

    def test_reshapes_and_adds_bias(self) -> None:
        w, b = torch.randn(8, 16), torch.randn(8)
        x = torch.randn(2, 3, 16)
        out = quantized_linear(_TOY, x, (w,), b, lambda x2d: _dequant(x2d, w))
        torch.testing.assert_close(out, F.linear(x, w, b))

    def test_kernel_path_when_launchable(self) -> None:
        w, x = torch.randn(8, 16), torch.randn(2, 16)
        with mock.patch.object(gd, "can_launch_triton", return_value=True):
            out = quantized_linear(_TOY, x, (w,), None, lambda x2d: self.fail("fell back"))
        torch.testing.assert_close(out, x @ w.t() + 2)


class ChunkedDequantLinearTest(unittest.TestCase):
    def test_small_weight_is_one_call(self) -> None:
        calls = []

        def rows(i, j):
            calls.append((i, j))
            return torch.zeros(1, j - i)

        chunked_dequant_linear(torch.randn(1, 4), 100, rows)
        self.assertEqual(calls, [(0, 100)])

    def test_large_weight_is_chunked_and_identical(self) -> None:
        x, w = torch.randn(2, 16), torch.randn(10, 16)
        calls = []

        def rows(i, j):
            calls.append((i, j))
            return F.linear(x, w[i:j])

        with mock.patch.object(gd, "_DEQUANT_N_THRESHOLD", 4), mock.patch.object(gd, "_DEQUANT_N_CHUNK", 3):
            out = chunked_dequant_linear(x, 10, rows)
        self.assertEqual(calls, [(0, 3), (3, 6), (6, 9), (9, 10)])
        torch.testing.assert_close(out, F.linear(x, w))


if __name__ == "__main__":
    unittest.main()
