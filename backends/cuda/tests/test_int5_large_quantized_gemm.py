# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""INT5 BF16 tile, split reduction, and export routing regression checks."""

import unittest

import torch
import triton
from executorch.backends.cuda.dp4a_planar_int5_tensor import (
    CudaDp4aPlanarInt5Tensor,
    pack_int5,
)
from executorch.backends.cuda.quantize_op_dispatch.int5_dispatch import (
    _dequant_matmul_int5,
)
from executorch.backends.cuda.triton.kernels.int5_quantized_gemm import (
    _launch_w5a16,
    INT5_QUANTIZED_GEMM,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class Int5LargeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.reduced_precision = (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        )
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    def tearDown(self):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            self.reduced_precision
        )

    def _weights(self, n, k, group_size=32):
        q = torch.randint(0, 32, (n, k), dtype=torch.uint8, device="cuda")
        ql, qh = pack_int5(q)
        scale = torch.randint(
            1, 255, (n, k // group_size), dtype=torch.uint8, device="cuda"
        )
        steps = torch.full((n, k // 256), 0.0001, dtype=torch.float16, device="cuda")
        return ql, qh, scale, steps, torch.full_like(scale, 16), torch.ones_like(steps)

    def test_direct_and_split_tiles_with_tail_dimensions(self):
        config = triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 64, "PIPELINE_STAGES": 1},
            num_warps=4,
            num_stages=1,
        )
        cases = (
            (7, 8, 129, 768, 1),
            (16, 16, 129, 768, 1),
            (16, 16, 129, 768, 2),
            (16, 16, 129, 1280, 4),
            (32, 32, 257, 512, 1),
            (64, 64, 257, 512, 2),
        )
        for m, bucket, n, k, split in cases:
            with self.subTest(m=m, bucket=bucket, n=n, k=k, split=split):
                weights = self._weights(n, k)
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                ref = _dequant_matmul_int5(x, *weights, 32)
                out = _launch_w5a16(
                    bucket, x, *weights, 32, split_k=split, config=config
                )
                torch.testing.assert_close(out, ref, rtol=0.012, atol=0.025)
                self.assertTrue(torch.isfinite(out).all())
                replay = _launch_w5a16(
                    bucket, x, *weights, 32, split_k=split, config=config
                )
                self.assertTrue(torch.equal(out, replay))

    def test_large_buckets_reject_group_size_64(self):
        x = torch.randn(16, 512, device="cuda", dtype=torch.bfloat16)
        self.assertFalse(
            INT5_QUANTIZED_GEMM.supports(16, x, *self._weights(129, 512, 64), 64)
        )
        self.assertTrue(
            INT5_QUANTIZED_GEMM.supports(16, x, *self._weights(129, 512), 32)
        )

    def test_m16_export_uses_fused_op(self):
        n, k = 129, 512
        weights = self._weights(n, k)
        weight = CudaDp4aPlanarInt5Tensor(
            *weights, block_size=[1, 32], shape=torch.Size([n, k])
        )
        head = torch.nn.Linear(k, n, bias=False, device="meta")
        head.weight = torch.nn.Parameter(weight, requires_grad=False)
        x = torch.randn(1, 16, k, device="cuda", dtype=torch.bfloat16)
        exported = torch.export.export(head, (x,), strict=True)
        self.assertIn("int5_quantized_gemm_m16", str(exported.graph))


if __name__ == "__main__":
    unittest.main()
