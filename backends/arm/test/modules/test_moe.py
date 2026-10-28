# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Test fixed-top-k MoE lowering with finite, distinct router scores.

Non-finite and equal router scores are covered by the TopK operator tests.

"""

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    VgfPipeline,
)
from executorch.examples.models.llama.llama_transformer import MOEFeedForward
from executorch.examples.models.llama.model_args import ModelArgs

input_t = tuple[torch.Tensor]

single_token = torch.tensor(
    [[-0.875, -0.625, -0.375, -0.125, 0.125, 0.375, 0.625, 0.875]],
    dtype=torch.float32,
)
three_tokens = torch.tensor(
    [
        [-0.875, -0.625, -0.375, -0.125, 0.125, 0.375, 0.625, 0.875],
        [0.625, 0.875, -0.875, -0.625, -0.375, -0.125, 0.125, 0.375],
        [0.125, 0.375, 0.625, 0.875, -0.875, -0.625, -0.375, -0.125],
    ],
    dtype=torch.float32,
)

test_data = {
    "t1_e8_k1_fp16": (single_token.half(), 8, 1),
    "t1_e8_k2_fp32": (single_token, 8, 2),
    "t1_e8_k4_fp16": (single_token.half(), 8, 4),
    "t3_e8_k1_fp32": (three_tokens, 8, 1),
    "t3_e8_k2_fp16": (three_tokens.half(), 8, 2),
    "t3_e8_k4_fp32": (three_tokens, 8, 4),
    "t3_e4_k4_fp16": (three_tokens.half(), 4, 4),
}


def _make_moe(experts: int, k: int, dtype: torch.dtype) -> MOEFeedForward:
    model = MOEFeedForward(
        ModelArgs(
            dim=8,
            hidden_dim=16,
            moe_hidden_dim=16,
            n_heads=2,
            num_experts=experts,
            num_activated_experts=k,
            moe_score_func="softmax",
            moe_use_expert_bias=False,
            moe_shared_expert_hidden_dim=None,
        )
    ).eval()
    generator = torch.Generator().manual_seed(2647)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.25, 0.25, generator=generator)
        # Keep distinct input values as router scores so TopK selection
        # is unambiguous.
        model.gate.weight.copy_(torch.eye(8)[:experts])
    return model.to(dtype)


@common.parametrize("test_data", test_data)
def test_moe_tosa_FP(test_data: tuple[torch.Tensor, int, int]) -> None:
    x, experts, k = test_data
    model = _make_moe(experts, k, x.dtype)
    pipeline = TosaPipelineFP[input_t](
        model,
        (x,),
        [
            "torch.ops.aten.topk.default",
            "torch.ops.aten.index.Tensor",
            "torch.ops.aten.einsum.default",
        ],
        "executorch_exir_dialects_edge__ops_",
        tosa_extensions=["INT"],
    )
    pipeline.run()


@common.parametrize("test_data", test_data)
@common.SkipIfNoModelConverter
def test_moe_vgf_no_quant(test_data: tuple[torch.Tensor, int, int]) -> None:
    x, experts, k = test_data
    model = _make_moe(experts, k, x.dtype)
    pipeline = VgfPipeline[input_t](
        model,
        (x,),
        [
            "torch.ops.aten.topk.default",
            "torch.ops.aten.index.Tensor",
            "torch.ops.aten.einsum.default",
        ],
        "executorch_exir_dialects_edge__ops_",
        quantize=False,
    )
    pipeline.run()
