# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Test fixed-top-k MoE lowering with finite, distinct router scores.

Equal router scores are covered by the TopK operator tests. Non-finite router
scores are outside the supported domain.

"""

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    VgfPipeline,
)
from executorch.examples.models.llama.llama_transformer import MOEFeedForward
from executorch.examples.models.llama.model_args import ModelArgs
from transformers import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts, GptOssMLP
from typing_extensions import override

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


# GPT-OSS expects [batch, sequence, hidden] inputs.
gpt_oss_test_data = {
    "fp32": (three_tokens.unsqueeze(0),),
    "fp16": (three_tokens.unsqueeze(0).half(),),
}


class TensorizedGptOssExperts(GptOssExperts):
    """Provide tensorized experts for the native GPT-OSS MLP.

    Used with GptOssMLP from Transformers 5.0.0rc1 and 5.3.0::

        GptOssMLP
        |-- router: GptOssTopKRouter
        `-- experts: TensorizedGptOssExperts

    Inherits expert parameters from GptOssExperts and reproduces its clipped
    activation. Only forward() is rewritten as gather/BMM/reduction.
    Accepts either 2D or 3D expert inputs and returns outputs with the same shape.

    """

    @override
    def forward(
        self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None
    ) -> torch.Tensor:
        tokens, k = router_indices.shape
        dim = self.hidden_size
        intermediate = self.intermediate_size
        input_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, dim)
        selected_up = self.gate_up_proj[router_indices]
        selected_down = self.down_proj[router_indices]
        expanded_x = hidden_states[:, None, :].expand(tokens, k, dim)
        gate_up = torch.bmm(
            expanded_x.reshape(tokens * k, 1, dim),
            selected_up.reshape(tokens * k, dim, 2 * intermediate),
        ).reshape(tokens, k, 2 * intermediate)
        gate_up = gate_up + self.gate_up_proj_bias[router_indices]
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        hidden = (up + 1) * glu
        outputs = torch.bmm(
            hidden.reshape(tokens * k, 1, intermediate),
            selected_down.reshape(tokens * k, intermediate, dim),
        ).reshape(tokens, k, dim)
        outputs = outputs + self.down_proj_bias[router_indices]
        outputs = (outputs * routing_weights.unsqueeze(-1)).sum(dim=1)
        return outputs.reshape(input_shape)


def _make_gpt_oss_moe(dtype: torch.dtype) -> tuple[GptOssMLP, GptOssConfig]:
    config = GptOssConfig(
        hidden_size=8,
        intermediate_size=16,
        num_local_experts=8,
        num_experts_per_tok=4,
        experts_implementation="eager",
    )
    model = GptOssMLP(config)
    model.experts = TensorizedGptOssExperts(config)
    generator = torch.Generator().manual_seed(2648)
    with torch.no_grad():
        for parameter in model.experts.parameters():
            parameter.uniform_(-0.25, 0.25, generator=generator)
        model.router.weight.copy_(torch.eye(model.router.weight.shape[1]))
        model.router.bias.copy_(torch.linspace(-0.07, 0.07, model.router.bias.numel()))
        # Exercise gate/up clipping and unclipped negative gates.
        model.experts.gate_up_proj_bias[:, :8].add_(
            torch.tensor([9.0, 0.5, 0.5, 9.0, 0.5, -9.0, -9.0, 0.5])
        )
    return model.eval().to(dtype), config


@common.parametrize("test_data", gpt_oss_test_data)
@torch.no_grad()
def test_tensorized_gpt_oss_moe_matches_reference(test_data: input_t) -> None:
    (x,) = test_data
    model, config = _make_gpt_oss_moe(x.dtype)
    reference = GptOssMLP(config).eval().to(x.dtype)
    reference.load_state_dict(model.state_dict())
    torch.testing.assert_close(model(x), reference(x))


@common.parametrize("test_data", gpt_oss_test_data)
@torch.no_grad()
def test_tensorized_gpt_oss_moe_tosa_FP(test_data: input_t) -> None:
    (x,) = test_data
    model, _ = _make_gpt_oss_moe(x.dtype)
    pipeline = TosaPipelineFP[input_t](
        model,
        test_data,
        [
            "torch.ops.aten.topk.default",
            "torch.ops.aten.index.Tensor",
            "torch.ops.aten.bmm.default",
            "torch.ops.aten.slice.Tensor",
            "torch.ops.aten.clamp.default",
        ],
        "executorch_exir_dialects_edge__ops_",
        tosa_extensions=["INT"],
    )
    pipeline.run()
