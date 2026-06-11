# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn.functional as F
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

        self.linear = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.lora_a = nn.Linear(in_features=in_dim, out_features=rank, bias=False)
        self.lora_b = nn.Linear(in_features=rank, out_features=out_dim, bias=False)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Remap keys to "linear.*"
        for attr in ("weight", "bias"):
            old_key = prefix + attr
            new_key = prefix + "linear." + attr
            if old_key in state_dict and new_key not in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        # Optional forward-arg LoRA tensors (CoreML LoRA-as-IO Path 2). When
        # both are provided, they override the stored lora_a/lora_b for this
        # call. Default behavior (None, None) is unchanged.
        lora_a: Optional[torch.Tensor] = None,
        lora_b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.linear(x)
        if lora_a is not None and lora_b is not None:
            z = F.linear(self.dropout(x), lora_a)
            z = (self.alpha / self.rank) * F.linear(z, lora_b)
        else:
            z = self.lora_a(self.dropout(x))
            z = (self.alpha / self.rank) * self.lora_b(z)
        return out + z
