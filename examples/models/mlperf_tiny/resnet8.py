# Copyright 2026 Arm Limited and/or its affiliates.
#
# SPDX-License-Identifier: Apache-2.0

"""
MLPerf Tiny CIFAR-10 ResNet-8 model for Arm backend testing.

Architecture follows the MLPerf Tiny benchmark specification:
three residual stages with channel widths [16, 32, 64] operating
on 32x32 input images, producing 10-class output logits.
"""

import torch

from executorch.examples.models.model_base import EagerModelBase
from torch import nn, Tensor

# Channel configuration for each residual stage.
_STAGE_CHANNELS = (16, 32, 64)


def _make_conv(ch_in: int, ch_out: int, ks: int, s: int = 1, pad: int = 0) -> nn.Conv2d:
    """Create a Conv2d layer without bias (used before batch-norm)."""
    return nn.Conv2d(ch_in, ch_out, kernel_size=ks, stride=s, padding=pad, bias=False)


class _ResStage(nn.Module):
    """Single residual stage: two 3x3 conv-bn pairs with a skip connection."""

    def __init__(self, ch_in: int, ch_out: int, downsample: bool = False) -> None:
        super().__init__()
        s = 2 if downsample else 1

        self.path_a = nn.Sequential(
            _make_conv(ch_in, ch_out, ks=3, s=s, pad=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            _make_conv(ch_out, ch_out, ks=3, s=1, pad=1),
            nn.BatchNorm2d(ch_out),
        )

        self.skip = (
            _make_conv(ch_in, ch_out, ks=1, s=s)
            if (downsample or ch_in != ch_out)
            else nn.Identity()
        )

        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.path_a(x) + self.skip(x))


class ResNet8(nn.Module):
    """
    Three-stage residual network for the MLPerf Tiny image-classification
    benchmark (CIFAR-10, 32x32 RGB, 10 classes).
    """

    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()

        # Initial convolution + normalisation.
        self.entry = nn.Sequential(
            _make_conv(3, _STAGE_CHANNELS[0], ks=3, s=1, pad=1),
            nn.BatchNorm2d(_STAGE_CHANNELS[0]),
            nn.ReLU(),
        )

        # Three residual stages; stages 2 and 3 halve spatial resolution.
        self.stages = nn.Sequential(
            _ResStage(_STAGE_CHANNELS[0], _STAGE_CHANNELS[0], downsample=False),
            _ResStage(_STAGE_CHANNELS[0], _STAGE_CHANNELS[1], downsample=True),
            _ResStage(_STAGE_CHANNELS[1], _STAGE_CHANNELS[2], downsample=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(_STAGE_CHANNELS[-1], n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.entry(x)
        x = self.stages(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


class ResNet8Model(EagerModelBase):
    def get_eager_model(self) -> nn.Module:
        return ResNet8().eval()

    def get_example_inputs(self):
        return (torch.rand(1, 3, 32, 32) * 2 - 1,)
