# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import itertools
import unittest

from typing import Optional, Sequence

import torch


class UpsampleNearest2dTest(unittest.TestCase):
    def run_upsample_test(
        self,
        inp: torch.Tensor,
        output_size: Optional[Sequence[int]] = None,
        scale_factors: Optional[Sequence[float]] = None,
        atol=1e-7,
    ) -> None:
        aten_result = torch.nn.functional.interpolate(
            inp,
            size=output_size,
            mode="nearest",
            scale_factor=scale_factors,
        )
        et_result = torch.zeros_like(aten_result)
        et_result = torch.ops.et_test.upsample_nearest2d(
            inp,
            output_size=output_size,
            scale_factors=scale_factors,
            out=et_result,
        )
        self.assertTrue(
            torch.allclose(et_result, aten_result, atol=atol),
            msg=f"ET: {et_result} \n ATen: {aten_result} \n Error: {et_result.to(torch.float) - aten_result.to(torch.float)}",
        )

    def test_upsample_nearest2d_aten_parity_f32(self):
        N = [1, 2]
        C = [1, 3]
        H = [1, 3, 50, 1001]
        W = [1, 2, 62, 1237]
        OUT_H = [5, 21]
        OUT_W = [7, 31]

        for n, c, h, w, out_h, out_w in itertools.product(N, C, H, W, OUT_H, OUT_W):
            input = torch.randn(n, c, h, w)
            self.run_upsample_test(input, output_size=(out_h, out_w))
            self.run_upsample_test(input, scale_factors=(out_h / h, out_w / w))

    def test_upsample_nearest2d_aten_parity_u8(self):
        N = [1, 2]
        C = [1, 3]
        H = [1, 3, 50, 1001]
        W = [1, 2, 62, 1237]
        OUT_H = [5, 21]
        OUT_W = [7, 31]

        for n, c, h, w, out_h, out_w in itertools.product(N, C, H, W, OUT_H, OUT_W):
            input = torch.randint(0, 255, (n, c, h, w), dtype=torch.uint8)
            self.run_upsample_test(input, output_size=(out_h, out_w), atol=1)
            self.run_upsample_test(
                input,
                scale_factors=(out_h / h, out_w / w),
                atol=2,
            )
