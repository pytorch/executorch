# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Optional

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestSDPA(unittest.TestCase):
    class SDPA(torch.nn.Module):
        def __init__(self, scale: Optional[float] = None):
            super().__init__()
            self.dropout_p: float = 0.0
            self.is_causal: bool = False
            self.scale = scale

        def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ):
            return torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout_p,
                is_causal=self.is_causal,
                scale=self.scale,
            )

        @staticmethod
        def get_input_tensors(mask_rank: int):
            batch_size = 8
            heads = 16
            seq_len = 32
            dim = 64

            q = torch.randn(batch_size, heads, seq_len, dim)
            k = torch.randn(batch_size, heads, seq_len, dim)
            v = torch.randn(batch_size, heads, seq_len, dim)

            mask = None
            if mask_rank > 0:
                assert mask_rank >= 2, "mask rank must be >= 2"
                mask = torch.full((seq_len, seq_len), 0, dtype=torch.float)
                while mask.ndim < mask_rank:
                    mask.unsqueeze_(0)

            return (q, k, v, mask)

    def _test(self, module, inputs):
        (
            Tester(module, inputs)
            .export()
            .to_edge()
            .check_count({"executorch_exir_dialects_edge__ops_aten_bmm_default": 2})
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                ["executorch_exir_dialects_edge__ops_aten_bmm_default"],
            )
            .to_executorch()
            .serialize()
            .run_method()
            .compare_outputs()
        )

    def test_fp32_sdpa_mask2d(self):
        """
        Tests that the SDPA operator is correctly lowered to XNNPACK
        """
        module = self.SDPA()
        inputs = module.get_input_tensors(mask_rank=2)
        self._test(module, inputs)

    def test_fp32_sdpa_userscale(self):
        """
        Tests that the scale parameter is passed correctly to the SDPA operator
        """
        module = self.SDPA(scale=0.1234)
        inputs = module.get_input_tensors(mask_rank=2)
        self._test(module, inputs)

    @unittest.expectedFailure
    def test_fp32_sdpa_nomask(self):
        module = self.SDPA()
        inputs = module.get_input_tensors(mask_rank=0)
        # AssertionError: SubgraphMatcher cannot be initialized with an pattern with dead code
        # This is from attn_mask=None arg
        self._test(module, inputs)

    @unittest.expectedFailure
    def test_fp32_sdpa_mask4d(self):
        """
        Tests that the scale parameter is passed correctly to the SDPA operator
        """
        module = self.SDPA(scale=0.1234)
        # can't mask.squeeze_(0) yet with xnnpack
        inputs = module.get_input_tensors(mask_rank=4)
        self._test(module, inputs)
