# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import conftest
import torch

from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
)


class SDPA(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )


input_t = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def test_sdpa_FP():
    test_input = tuple(torch.randn(1, 3, 197, 64) for x in range(3))
    pipeline = TosaPipelineFP[input_t](
        SDPA(), 
        test_input, 
        [], 
        [],
        run_on_tosa_ref_model=conftest.is_option_enabled("tosa_ref_model"),
    )
    pipeline.pop_stage("check_count.exir")
    pipeline.run()


def test_sdpa_INT():
    test_input = tuple(torch.randn(1, 3, 197, 64) for x in range(3))
    pipeline = TosaPipelineINT[input_t](
        SDPA(), 
        test_input, 
        [], 
        [],
        run_on_tosa_ref_model=conftest.is_option_enabled("tosa_ref_model"),
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.pop_stage("check_count.exir")
    pipeline.run()
