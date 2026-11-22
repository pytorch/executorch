# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mixture-of-experts cases from transformers' own modules with its batched_mm
experts implementation: each token's routed experts' weights are gathered by
expert id and multiplied in a batched matmul. Qwen3.5 MoE's sparse block (a
softmax router renormalized over its top k, SiLU experts and a sigmoid-gated
shared expert) and Gemma 4's (a normalized, scaled router with per-expert
scales and GELU-tanh experts). Every case holds stacked expert weights, which
weight-only schemes quantize, so they are named quantized_*."""

import torch
from executorch.backends.native.test.samples.cases import Case, draws
from transformers.models.gemma4 import modeling_gemma4 as gemma4
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qwen3_5_moe
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)


def _init_experts(experts: torch.nn.Module) -> torch.nn.Module:
    # transformers leaves stacked expert weights uninitialized.
    for weight in (experts.gate_up_proj, experts.down_proj):
        torch.nn.init.normal_(weight, std=0.05)
    return experts


def _qwen35_config(experts: int, top_k: int, shared: int) -> Qwen3_5MoeTextConfig:
    # Every weight row is a multiple of 256, so GGUF's K-quants fit.
    config = Qwen3_5MoeTextConfig(
        hidden_size=256,
        num_experts=experts,
        num_experts_per_tok=top_k,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=shared,
    )
    config._experts_implementation = "batched_mm"
    return config


class _Experts(torch.nn.Module):
    """Qwen3.5 MoE's experts alone, the routing an input: expert ids and their
    weights per token."""

    def __init__(self, config: Qwen3_5MoeTextConfig) -> None:
        super().__init__()
        self.experts = _init_experts(qwen3_5_moe.Qwen3_5MoeExperts(config))

    def forward(self, x, ids, weights):
        return self.experts(x, ids, weights)


def _experts_case(name: str, experts: int, top_k: int, tokens: int) -> Case:
    config = _qwen35_config(experts, top_k, 256)
    inputs = []
    for x, scores in draws(2, (tokens, 256), (tokens, experts)):
        weights, ids = scores.softmax(-1).topk(top_k, dim=-1)
        inputs.append((x, ids, weights / weights.sum(-1, keepdim=True)))
    return Case(name, _Experts(config), inputs)


class _FloatLinear(torch.nn.Module):
    """An nn.Linear's weight applied by F.linear, which schemes leave in
    float, as GGUF files keep MoE routers and gates."""

    def __init__(self, linear: torch.nn.Linear) -> None:
        super().__init__()
        self.weight = linear.weight

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight)


class _Qwen35Block(torch.nn.Module):
    def __init__(self, config: Qwen3_5MoeTextConfig) -> None:
        super().__init__()
        self.block = qwen3_5_moe.Qwen3_5MoeSparseMoeBlock(config)
        # GGUF files keep the shared expert's one-row gate in float too.
        self.block.shared_expert_gate = _FloatLinear(self.block.shared_expert_gate)
        _init_experts(self.block.experts)
        torch.nn.init.normal_(self.block.gate.weight, std=0.2)

    def forward(self, x):
        return self.block(x)


def _qwen35_block_case(name: str, experts: int, top_k: int, tokens: int) -> Case:
    block = _Qwen35Block(_qwen35_config(experts, top_k, 256))
    return Case(name, block, draws(2, (1, tokens, 256)))


class _Gemma4Block(torch.nn.Module):
    """Gemma 4's router and experts as its decoder layer calls them, the
    router in float."""

    def __init__(self, config: Gemma4TextConfig) -> None:
        super().__init__()
        self.router = gemma4.Gemma4TextRouter(config)
        # A quantized router's rounding can flip a near-tie between experts.
        self.router.proj = _FloatLinear(self.router.proj)
        self.experts = _init_experts(gemma4.Gemma4TextExperts(config))
        torch.nn.init.uniform_(self.router.scale, 0.5, 1.5)
        torch.nn.init.uniform_(self.router.per_expert_scale, 0.5, 1.5)

    def forward(self, x):
        _, weights, ids = self.router(x)
        return self.experts(x, ids, weights)


def _gemma4_block_case(name: str, experts: int, top_k: int, tokens: int) -> Case:
    config = Gemma4TextConfig(
        hidden_size=256,
        num_experts=experts,
        top_k_experts=top_k,
        moe_intermediate_size=256,
        hidden_activation="gelu_pytorch_tanh",
        enable_moe_block=True,
    )
    config._experts_implementation = "batched_mm"
    return Case(name, _Gemma4Block(config), draws(2, (tokens, 256)))


def cases() -> list[Case]:
    torch.manual_seed(0)
    # Routing weights scale all of a token's outputs together, so its rounding
    # biases them alike; enough tokens average those biases out.
    return [
        _experts_case("quantized_experts_top1", experts=4, top_k=1, tokens=32),
        _experts_case("quantized_experts_top2", experts=4, top_k=2, tokens=32),
        _experts_case("quantized_experts_top8", experts=16, top_k=8, tokens=16),
        _qwen35_block_case("quantized_qwen35_block", experts=8, top_k=2, tokens=32),
        _gemma4_block_case("quantized_gemma4_block", experts=8, top_k=2, tokens=32),
    ]
