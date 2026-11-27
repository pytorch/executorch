# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

DEFAULT_BACKBONE_CHANNELS = (64, 128, 128, 128, 128, 64, 64, 64)
DEFAULT_HIDDEN_FEATURES = 1024
DEFAULT_GRID_SIZE = 7
DEFAULT_NUM_BOXES = 1


class ChannelMask(torch.nn.Module):
    """Channel mask used for pruning in MicroYolo."""

    mask: torch.Tensor

    def __init__(self, channels: int):
        super().__init__()
        self.register_buffer("mask", torch.ones(channels))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.mask.reshape(1, -1, *([1] * (inputs.ndim - 2)))

    @property
    def active_indices(self) -> torch.Tensor:
        return self.mask.nonzero(as_tuple=True)[0]


class DepthwiseSeparableConv(torch.nn.Sequential):
    """Base block for MicroYolo."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super().__init__(
            torch.nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride,
                padding,
                groups=in_channels,
            ),
            torch.nn.Conv2d(in_channels, out_channels, 1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )


class MicroYolo(torch.nn.Module):
    """
    Model architecture based on μYOLO: Towards Single-Shot Object Detection
    on Microcontrollers (https://arxiv.org/pdf/2408.15865) with masking based pruning.
    """

    example_input: torch.Tensor = torch.ones(1, 3, 128, 128)

    def __init__(
        self,
        grid_size: int = DEFAULT_GRID_SIZE,
        num_boxes: int = DEFAULT_NUM_BOXES,
        backbone_channels: tuple[int, ...] = DEFAULT_BACKBONE_CHANNELS,
        hidden_features: int = DEFAULT_HIDDEN_FEATURES,
        pruning_masks: bool = True,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.num_classes = 1
        self.num_boxes = num_boxes
        self.backbone_channels = tuple(backbone_channels)
        self.hidden_features = hidden_features
        self.pruning_masks = pruning_masks
        output_features = grid_size * grid_size * num_boxes * 5
        stem, *separable = self.backbone_channels

        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, stem, 4, 2),
            torch.nn.BatchNorm2d(stem),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            DepthwiseSeparableConv(stem, separable[0], 3, 1, 0),
            DepthwiseSeparableConv(separable[0], separable[1], 3, 1, 1),
            DepthwiseSeparableConv(separable[1], separable[2], 3, 1, 0),
            torch.nn.MaxPool2d(2),
            DepthwiseSeparableConv(separable[2], separable[3], 3, 1, 1),
            DepthwiseSeparableConv(separable[3], separable[4], 3, 1, 0),
            DepthwiseSeparableConv(separable[4], separable[5], 3, 1, 1),
            DepthwiseSeparableConv(separable[5], separable[6], 3, 1, 0),
            torch.nn.MaxPool2d(2),
        )

        self.hidden_mask: torch.nn.Module
        if pruning_masks:
            self.feature_masks = torch.nn.ModuleList(
                ChannelMask(channels) for channels in self.backbone_channels
            )
            self.hidden_mask = ChannelMask(hidden_features)
        else:
            self.feature_masks = torch.nn.ModuleList(
                torch.nn.Identity() for _ in self.backbone_channels
            )
            self.hidden_mask = torch.nn.Identity()

        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.backbone_channels[-1] * 4 * 4, hidden_features),
            torch.nn.BatchNorm1d(hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features, output_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask_index = 0
        for layer in self.backbone:
            if isinstance(layer, DepthwiseSeparableConv):
                x = layer[0](x)
                x = self.feature_masks[mask_index](x)
                x = layer[1](x)
                x = layer[2](x)
                x = layer[3](x)
                mask_index += 1
                x = self.feature_masks[mask_index](x)
            else:
                x = layer(x)
            if isinstance(layer, torch.nn.ReLU):
                x = self.feature_masks[mask_index](x)

        x = torch.flatten(x, 1)
        x = self.head[0](x)
        x = self.head[1](x)
        x = self.head[2](x)
        x = self.hidden_mask(x)
        return self.head[3](x)


def model_from_checkpoint(checkpoint: dict) -> MicroYolo:
    architecture = checkpoint.get("architecture", {})
    backbone_channels = tuple(
        architecture.get("backbone_channels", DEFAULT_BACKBONE_CHANNELS)
    )
    hidden_features = architecture.get("hidden_features", DEFAULT_HIDDEN_FEATURES)
    return MicroYolo(
        grid_size=checkpoint["grid_size"],
        num_boxes=checkpoint["num_boxes"],
        backbone_channels=backbone_channels,
        hidden_features=hidden_features,
        pruning_masks=architecture.get("pruning_masks", True),
    )
