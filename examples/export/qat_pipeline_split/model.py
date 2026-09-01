# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Model definition for the QAT pipeline split example.
#
# SmallConvNet is used only by stage 1 (1_prepare.py), which constructs the
# eager model before capture.  Stages 2 and 3 reload the model from a saved
# .pt2 file and only import the example-input helpers below.

from typing import List, Tuple

import torch
import torch.nn as nn


class SmallConvNet(nn.Module):
    """A small conv net for demonstrating PT2E quantization.

    Input:  (N, 1, 28, 28)
    Output: (N, 10)
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(8 * 4 * 4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def get_model() -> SmallConvNet:
    return SmallConvNet().eval()


def get_example_inputs() -> Tuple[torch.Tensor, ...]:
    return (torch.randn(1, 1, 28, 28),)


def get_calibration_inputs(n: int = 4) -> List[Tuple[torch.Tensor, ...]]:
    return [(torch.randn(1, 1, 28, 28),) for _ in range(n)]
