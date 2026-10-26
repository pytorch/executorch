# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyModel(nn.Module):
    """Plain 4-layer toy model, randomly initialized (no checkpoint loading)."""

    def __init__(self, in_dim: int = 8, hidden_dim: int = 32, out_dim: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc4 = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return torch.tanh(self.fc4(x))


class ToyModelWithLoss(nn.Module):
    """Wraps ToyModel with the loss embedded, as required for on-device training.

    MSE (not cross-entropy) keeps the joint-graph decomposition table minimal
    -- no ``nll_loss_backward`` to handle.
    """

    def __init__(self, toy_model: ToyModel):
        super().__init__()
        self.toy_model = toy_model

    def forward(self, x, target):
        pred = self.toy_model(x)
        return F.mse_loss(pred, target).reshape(1), pred.detach()
