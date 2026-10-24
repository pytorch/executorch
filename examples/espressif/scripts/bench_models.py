# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Model factories for examples/espressif/scripts/benchmark.py.

Use with aot_espnn_compiler.py as --model examples/espressif/scripts/bench_models.py:<factory>.
Small models use fixed-seed weights: they measure latency and memory, not accuracy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _randomize_batch_norm(model: nn.Module, seed: int) -> nn.Module:
    generator = torch.Generator().manual_seed(seed)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            channels = module.num_features
            module.running_mean.copy_(torch.randn(channels, generator=generator) * 0.1)
            module.running_var.copy_(
                torch.rand(channels, generator=generator) * 0.5 + 0.75
            )
            module.weight.data.copy_(
                torch.rand(channels, generator=generator) * 0.5 + 0.75
            )
            module.bias.data.copy_(torch.randn(channels, generator=generator) * 0.1)
    return model


class DSCNN(nn.Module):
    """Keyword spotting DS-CNN (MLPerf Tiny style) on a 1x49x10 MFCC input."""

    def __init__(self, num_classes: int = 12, width: int = 64):
        super().__init__()
        layers = [
            nn.Conv2d(1, width, (10, 4), stride=(2, 2), padding=(5, 1), bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(),
        ]
        for _ in range(4):
            layers += [
                nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(),
                nn.Conv2d(width, width, 1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(),
            ]
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(width, num_classes)

    def forward(self, x):
        return self.classifier(torch.flatten(self.pool(self.features(x)), 1))


class BasicBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(y)) + x)


class TinyResNet(nn.Module):
    """64x64 image classifier with residuals, grouped conv and concat."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.block1 = BasicBlock(24)
        self.down = nn.Sequential(
            nn.Conv2d(24, 48, 3, stride=2, padding=1, groups=4, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU6(),
        )
        self.branch_a = nn.Sequential(nn.Conv2d(48, 32, 1), nn.ReLU())
        self.branch_b = nn.Sequential(nn.Conv2d(48, 32, 3, padding=1), nn.ReLU())
        self.block2 = BasicBlock(64)
        self.avg = nn.AvgPool2d(2)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.block1(self.pool(self.stem(x)))
        x = self.down(x)
        x = self.block2(torch.cat([self.branch_a(x), self.branch_b(x)], dim=1))
        return self.head(self.avg(x))


class TinyTransformer(nn.Module):
    """Patch-embedding transformer: shows the cost of CPU fallback (LayerNorm, attention)."""

    def __init__(self, dim: int = 32, heads: int = 2, classes: int = 10):
        super().__init__()
        self.heads = heads
        self.stem = nn.Conv2d(3, dim, 8, stride=8)
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp1 = nn.Linear(dim, 2 * dim)
        self.mlp2 = nn.Linear(2 * dim, dim)
        self.head = nn.Linear(dim, classes)

    def forward(self, x):
        tokens = self.stem(x).flatten(2).transpose(1, 2)
        n, t, d = tokens.shape
        q, k, v = self.qkv(self.norm1(tokens)).chunk(3, dim=-1)
        q, k, v = (
            z.reshape(n, t, self.heads, d // self.heads).transpose(1, 2)
            for z in (q, k, v)
        )
        attention = torch.softmax(
            q @ k.transpose(-2, -1) / (d // self.heads) ** 0.5, dim=-1
        )
        tokens = tokens + self.proj((attention @ v).transpose(1, 2).reshape(n, t, d))
        tokens = tokens + self.mlp2(F.gelu(self.mlp1(self.norm2(tokens))))
        return self.head(tokens.mean(dim=1))


class SegmentationOut(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]


class DetectorHeads(nn.Module):
    """SSDLite backbone and heads; anchor decoding and NMS stay on the host or app side."""

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head

    def forward(self, x):
        head = self.head(list(self.backbone(x).values()))
        return head["bbox_regression"], head["cls_logits"]


def ds_cnn_kws() -> nn.Module:
    torch.manual_seed(1)
    return _randomize_batch_norm(DSCNN(), 1).eval()


def tiny_resnet() -> nn.Module:
    torch.manual_seed(0)
    return _randomize_batch_norm(TinyResNet(), 0).eval()


def tiny_transformer() -> nn.Module:
    torch.manual_seed(2)
    return TinyTransformer().eval()


def lraspp_mobilenet_v3() -> nn.Module:
    import torchvision

    return SegmentationOut(
        torchvision.models.segmentation.lraspp_mobilenet_v3_large(weights="DEFAULT")
    ).eval()


def ssdlite320_mobilenet_v3() -> nn.Module:
    import torchvision

    return DetectorHeads(
        torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights="DEFAULT")
    ).eval()
