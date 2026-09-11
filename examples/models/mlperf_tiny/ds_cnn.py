# Copyright 2026 Arm Limited and/or its affiliates.
#
# SPDX-License-Identifier: Apache-2.0

"""PyTorch reimplementation of the MLCommons Tiny Keyword Spotting DS-CNN model."""

import torch
import torch.nn as nn

from executorch.examples.models.model_base import EagerModelBase


class DepthwiseSeparableConv(nn.Module):
    """Applies a depthwise convolution followed by a pointwise projection."""

    def __init__(self, channels: int, kernel_size: tuple[int, int] = (3, 3)) -> None:
        super().__init__()
        padding = tuple(k // 2 for k in kernel_size)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=True,
        )
        self.depthwise_bn = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.pointwise_bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.depthwise_bn(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.pointwise_bn(x)
        x = self.relu(x)
        return x


class DSCNNKWS(nn.Module):
    """MLCommons Tiny keyword spotting model; returns logits unless apply_softmax is set."""

    def __init__(self, num_classes: int = 12, *, apply_softmax: bool = False) -> None:
        super().__init__()
        self.apply_softmax = apply_softmax
        self.num_classes = num_classes
        self.input_height = 49
        self.input_width = 10
        self.input_channels = 1
        # Implements SAME padding for stride 2 kernel.
        self.input_padding = nn.ZeroPad2d((1, 1, 4, 5))
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=64,
                kernel_size=(10, 4),
                stride=(2, 2),
                padding=0,
                bias=True,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            DepthwiseSeparableConv(64),
            DepthwiseSeparableConv(64),
            DepthwiseSeparableConv(64),
            DepthwiseSeparableConv(64),
            nn.Dropout(p=0.4),
        )
        self.pool = nn.AvgPool2d(kernel_size=(25, 5))
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_padding(x)
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return torch.softmax(x, dim=-1) if self.apply_softmax else x


class DSCNNKWSModel(EagerModelBase):

    def get_eager_model(self) -> torch.nn.Module:
        return DSCNNKWS(apply_softmax=True).eval()

    def get_example_inputs(self):
        return (torch.rand(1, 1, 49, 10) * 2 - 1,)
