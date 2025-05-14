# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch

from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineBI,
    TosaPipelineMI,
)


class SDPA(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )


input_t = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def test_sdpa_MI():
    test_input = tuple(torch.randn(1, 3, 197, 64) for x in range(3))
    pipeline = TosaPipelineMI[input_t](SDPA(), test_input, [], [])
    pipeline.pop_stage("check_count.exir")
    pipeline.run()


def test_sdpa_BI():
    test_input = tuple(torch.randn(1, 3, 197, 64) for x in range(3))
    pipeline = TosaPipelineBI[input_t](SDPA(), test_input, [], [])
    pipeline.pop_stage("check.quant_nodes")
    pipeline.pop_stage("check_count.exir")
    pipeline.pop_stage(
        "run_method_and_compare_outputs"
    )  # TODO: reference is not quantized
    pipeline.run()
