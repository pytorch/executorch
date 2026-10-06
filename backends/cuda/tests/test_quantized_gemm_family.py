# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for QuantizedGemmFamily (triton/kernels/quantized_gemm_family.py).

    python -m pytest backends/cuda/tests/test_quantized_gemm_family.py -v
"""

import unittest

import torch
from executorch.backends.cuda.triton.kernels.quantized_gemm_family import (
    QuantizedGemmFamily,
)

_CALLS: list[tuple[int, int]] = []


def _prototype(x: torch.Tensor, weight: torch.Tensor, scale: float) -> torch.Tensor:
    raise NotImplementedError


def _launch(bucket: int, x: torch.Tensor, weight: torch.Tensor, scale: float) -> torch.Tensor:
    _CALLS.append((bucket, x.shape[0]))
    return (x @ weight.t()) * scale + bucket


def _fake(bucket: int, x: torch.Tensor, weight: torch.Tensor, scale: float) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]))


def _unsupported_reason(bucket: int, x: torch.Tensor, weight: torch.Tensor, scale: float):
    if x.shape[0] > bucket:
        return f"{x.shape[0]} rows exceed the bucket"
    if scale <= 0:
        return "scale must be positive"
    return None


_TOY = QuantizedGemmFamily(
    "quantized_gemm_family_test_toy", (4, 1, 2), _prototype, _launch, _fake, _unsupported_reason
)


class QuantizedGemmFamilyTest(unittest.TestCase):
    def test_registers_one_op_per_bucket_with_the_prototype_schema(self) -> None:
        self.assertEqual(_TOY.buckets, (1, 2, 4))
        for bucket in _TOY.buckets:
            op = getattr(torch.ops.triton, f"quantized_gemm_family_test_toy_m{bucket}")
            self.assertIn("Tensor x, Tensor weight, float scale", str(op.default._schema))
            self.assertIs(_TOY.op(bucket), _TOY.op(bucket))
        with self.assertRaisesRegex(RuntimeError, "unsupported"):
            _TOY.op(3)

    def test_op_validates_then_runs_its_bucket_launcher(self) -> None:
        _CALLS.clear()
        x, w = torch.randn(2, 16), torch.randn(8, 16)
        out = _TOY.op(4)(x, w, 2.0)
        self.assertEqual(_CALLS, [(4, 2)])
        torch.testing.assert_close(out, x @ w.t() * 2.0 + 4)
        with self.assertRaisesRegex(RuntimeError, "m1: 2 rows exceed the bucket"):
            _TOY.op(1)(x, w, 2.0)
        self.assertEqual(_CALLS, [(4, 2)])

    def test_supports_forwards_and_never_raises(self) -> None:
        x, w = torch.randn(2, 16), torch.randn(8, 16)
        self.assertTrue(_TOY.supports(2, x, w, 1.0))
        self.assertFalse(_TOY.supports(1, x, w, 1.0))
        self.assertFalse(_TOY.supports(2, x, w, -1.0))
        self.assertFalse(_TOY.supports(3, x, w, 1.0))

    def test_validate_raises_the_reason(self) -> None:
        x, w = torch.randn(2, 16), torch.randn(8, 16)
        _TOY.validate(2, x, w, 1.0)
        with self.assertRaisesRegex(RuntimeError, "scale must be positive"):
            _TOY.validate(2, x, w, -1.0)
        with self.assertRaisesRegex(RuntimeError, "unsupported .* bucket 3"):
            _TOY.validate(3, x, w, 1.0)

    def test_prototype_with_postponed_annotations_from_its_own_module(self) -> None:
        # The annotation names a type only the prototype's module knows.
        namespace: dict = {}
        exec(
            "from __future__ import annotations\n"
            "from torch import Tensor as OwnTensor\n"
            "def prototype(x: OwnTensor, weight: OwnTensor) -> OwnTensor: ...\n",
            namespace,
        )
        family = QuantizedGemmFamily(
            "quantized_gemm_family_test_postponed",
            (1,),
            namespace["prototype"],
            lambda bucket, x, w: x @ w.t(),
            lambda bucket, x, w: x.new_empty((x.shape[0], w.shape[0])),
            lambda bucket, x, w: None,
        )
        schema = str(torch.ops.triton.quantized_gemm_family_test_postponed_m1.default._schema)
        self.assertIn("Tensor x, Tensor weight", schema)
        x, w = torch.randn(1, 4), torch.randn(3, 4)
        torch.testing.assert_close(family.op(1)(x, w), x @ w.t())

    def test_rejects_invalid_buckets(self) -> None:
        for buckets in ((), (0, 1), (1, 1)):
            with self.assertRaises(ValueError):
                QuantizedGemmFamily(
                    "quantized_gemm_family_test_bad",
                    buckets,
                    _prototype,
                    _launch,
                    _fake,
                    _unsupported_reason,
                )


if __name__ == "__main__":
    unittest.main()
