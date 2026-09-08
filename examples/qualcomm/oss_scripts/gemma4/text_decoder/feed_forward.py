# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn


class Gemma4MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def prepare_feedforward_conv(self):
        self.gate_proj_conv = nn.Conv2d(
            self.hidden_size, self.intermediate_size, 1, bias=False
        )
        self.up_proj_conv = nn.Conv2d(
            self.hidden_size, self.intermediate_size, 1, bias=False
        )
        self.down_proj_conv = nn.Conv2d(
            self.intermediate_size, self.hidden_size, 1, bias=False
        )

        self.forward_no_conv = self.forward
        self.forward = self.forward_feedforward_conv

        self.gate_proj_conv.weight.data.copy_(self.gate_proj.weight[:, :, None, None])
        self.up_proj_conv.weight.data.copy_(self.up_proj.weight[:, :, None, None])
        self.down_proj_conv.weight.data.copy_(self.down_proj.weight[:, :, None, None])

        del self.gate_proj
        del self.up_proj
        del self.down_proj

    def forward_feedforward_conv(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.size()
        x = torch.reshape(x, (bsz, seq_len, 1, self.hidden_size)).transpose(1, 3)
        x = self.down_proj_conv(
            F.gelu(self.gate_proj_conv(x), approximate="tanh") * self.up_proj_conv(x)
        )
        x = x.transpose(1, 3)
        x = torch.reshape(x, (bsz, seq_len, -1))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )
