# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn


class LoRALinear(nn.Module):
    """LoRA linear layer as introduced in `LoRA: Low-Rank Adaptation of Large Language Models <https://arxiv.org/abs/2106.09685>`."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        use_bias: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha
        self.use_bias = use_bias
        self.dropout = dropout

        linear = nn.Linear(in_dim, out_dim, bias=use_bias)
        weight = linear.weight
        bias = linear.bias if self.use_bias else None
        self.register_parameter("weight", nn.Parameter(weight))
        self.register_parameter(
            "bias", nn.Parameter(bias) if bias is not None else None
        )

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.lora_a = nn.Linear(in_features=in_dim, out_features=rank, bias=False)
        self.lora_b = nn.Linear(in_features=rank, out_features=out_dim, bias=False)

    def merge(self) -> nn.Linear:
        """Merge LoRA weights into base weight, returning a standard nn.Linear.

        W_merged = W + (alpha / rank) * B @ A
        This eliminates the LoRA path at inference with zero additional latency.
        """
        merged = nn.Linear(self.in_dim, self.out_dim, bias=self.use_bias)
        merged.weight.data = self.weight + (self.alpha / self.rank) * (
            self.lora_b.weight @ self.lora_a.weight
        )
        if self.use_bias:
            merged.bias.data = self.bias.data.clone()
        return merged

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        lora_out = self.lora_a(self.dropout(x))
        lora_out = (self.alpha / self.rank) * self.lora_b(lora_out)

        return out + lora_out


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Replace all LoRALinear modules in the model with merged nn.Linear modules.

    Walks the module tree and substitutes each LoRALinear with a standard
    nn.Linear whose weight is W + (alpha/rank) * B @ A. This eliminates
    LoRA overhead at inference time.
    """
    for name, module in model.named_modules():
        for attr_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                merged = child.merge()
                setattr(module, attr_name, merged)
    return model
