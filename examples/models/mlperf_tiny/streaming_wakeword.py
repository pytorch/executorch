# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Incremental implementation of the MLPerf Tiny Streaming Wakeword model."""

from typing import cast

import torch

from executorch.examples.models.model_base import EagerModelBase
from torch import nn


class StreamingDepthwiseSeparableConv(nn.Module):
    history: torch.Tensor

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.register_buffer("history", torch.zeros(1, in_channels, kernel_size - 1, 1))
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            (kernel_size, 1),
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        window = torch.cat((self.history, frame), dim=2)
        result = self.dropout(
            self.relu(self.bn(self.pointwise(self.depthwise(window))))
        )
        self.history.copy_(window[:, :, 1:, :])
        return result


class StreamingWakeWord(nn.Module):
    """Consume one LFBE frame per call; outputs are valid after 30 frames.

    Classes are wakeword (Marvin), silence, and other, in that order.
    Call reset() before an independent stream. The first 29 outputs after
    reset are warm-up outputs, including when the stream ends earlier.
    """

    FRAME_SHAPE = (1, 40, 1, 1)
    RECEPTIVE_FIELD = 30
    HISTORY_NAMES = tuple(f"blocks.{i}.history" for i in range(4))

    def __init__(self, *, apply_softmax: bool = False):
        super().__init__()
        self.apply_softmax = apply_softmax
        self.blocks = nn.ModuleList(
            [
                StreamingDepthwiseSeparableConv(40, 128, 3),
                StreamingDepthwiseSeparableConv(128, 128, 5),
                StreamingDepthwiseSeparableConv(128, 128, 10),
                StreamingDepthwiseSeparableConv(128, 32, 15),
            ]
        )
        self.classifier = nn.Linear(32, 3)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.shape != self.FRAME_SHAPE:
            raise ValueError(f"Expected an LFBE frame of shape {self.FRAME_SHAPE}")
        for block in self.blocks:
            frame = block(frame)
        logits = self.classifier(frame.flatten(1))
        return torch.softmax(logits, dim=-1) if self.apply_softmax else logits

    def reset(self) -> None:
        for block in self.blocks:
            cast(torch.Tensor, block.history).zero_()


class StreamingWakeWordModel(EagerModelBase):
    def get_eager_model(self) -> nn.Module:
        return StreamingWakeWord(apply_softmax=True).eval()

    def get_example_inputs(self):
        return (torch.rand(StreamingWakeWord.FRAME_SHAPE) * 2 - 1,)
