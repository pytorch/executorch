# Copyright 2026 Arm Limited and/or its affiliates.
#
# SPDX-License-Identifier: Apache-2.0

"""PyTorch port of the MLPerf Tiny Visual Wake Words MobileNetV1 (width 0.25)."""

import torch
import torch.nn as nn

from executorch.examples.models.model_base import EagerModelBase


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.depthwise_bn(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.pointwise_bn(x)
        x = self.relu(x)
        return x


class MobileNetV1025(nn.Module):
    """MobileNetV1 with width multiplier 0.25 for the Visual Wake Words benchmark."""

    def __init__(self) -> None:
        super().__init__()
        self.num_classes = 2
        stem_out = 8
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_out, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_out),
            nn.ReLU(inplace=True),
        )

        channels = stem_out
        blocks = []
        cfg = [
            (1, True),
            (2, True),
            (1, False),
            (2, True),
            (1, False),
            (2, True),
            (1, False),
            (1, False),
            (1, False),
            (1, False),
            (1, False),
            (2, True),
            (1, False),
        ]
        for stride, expand in cfg:
            out_channels = channels * 2 if expand else channels
            blocks.append(DepthwiseSeparableConv(channels, out_channels, stride=stride))
            channels = out_channels

        self.features = nn.Sequential(*blocks)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MobileNetV1025Model(EagerModelBase):

    def get_eager_model(self) -> torch.nn.Module:
        return MobileNetV1025().eval()

    def get_example_inputs(self):
        return (torch.rand(1, 3, 96, 96) * 2 - 1,)
