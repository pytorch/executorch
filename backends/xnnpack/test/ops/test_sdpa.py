# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester

SAFE_SOFTMAX_GUARD_OPS = [
    "executorch_exir_dialects_edge__ops_aten_eq_Scalar",
    "executorch_exir_dialects_edge__ops_aten_any_dim",
    "executorch_exir_dialects_edge__ops_aten_logical_not_default",
]


class TestSDPA(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    class SDPA(torch.nn.Module):
        def __init__(self, is_causal: bool = False):
            super().__init__()
            self.is_causal = is_causal

        def forward(self, q, k, v, mask=None):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=self.is_causal
            )

    def _qkv(self):
        return tuple(torch.randn(1, 4, 16, 32) for _ in range(3))

    def test_fp32_sdpa_without_mask_drops_safe_softmax_guard(self):
        for is_causal in (False, True):
            with self.subTest(is_causal=is_causal):
                (
                    Tester(self.SDPA(is_causal), self._qkv())
                    .export()
                    .to_edge_transform_and_lower()
                    .check_not(
                        SAFE_SOFTMAX_GUARD_OPS
                        + ["executorch_exir_dialects_edge__ops_aten__softmax_default"]
                    )
                    .to_executorch()
                    .serialize()
                    .run_method_and_compare_outputs()
                )

    def test_fp32_sdpa_with_mask_keeps_safe_softmax_guard(self):
        mask = torch.zeros(16, 16)
        mask[0] = float("-inf")
        (
            Tester(self.SDPA(), (*self._qkv(), mask))
            .export()
            .to_edge_transform_and_lower()
            .check(SAFE_SOFTMAX_GUARD_OPS)
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )
