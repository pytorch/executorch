#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Toy Conv Model model for reproducing Vulkan backend device-specific issues.

Usage:
    python cnn_toy.py -o /tmp/cnn_toy
    python cnn_toy.py -o /tmp/cnn_toy --fp16
"""

import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import export

from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
    VulkanPartitioner,
)
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

# Approximate class counts matching Conv Model
N_TOPICS = 100
N_SUBTOPICS = 100
N_CONCEPTS = 120
N_KEYWORDS = 500
N_DISTASTEFUL = 50


class InvertedResidual(nn.Module):
    """MetaNet-style inverted residual block.

    Differs from standard MobileNetV2: no BN/activation after depthwise conv.

    1. Pointwise expansion (1x1 conv + BN + ReLU), skipped if expand_ratio=1
    2. Depthwise conv (kxk), no BN/activation
    3. Pointwise projection (1x1 conv + BN), no activation
    4. Residual add when stride=1 and in_ch == out_ch
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        stride=1,
        expand_ratio=1,
        kernel_size=3,
        group_size=None,
    ):
        super().__init__()
        self.use_residual = stride == 1 and in_ch == out_ch
        mid_ch = in_ch * expand_ratio
        padding = kernel_size // 2

        # group_size controls depthwise grouping: None means fully depthwise,
        # otherwise groups = mid_ch // group_size
        if group_size is not None:
            dw_groups = max(1, mid_ch // group_size)
        else:
            dw_groups = mid_ch

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_ch, mid_ch, 1, bias=False))
            layers.append(nn.BatchNorm2d(mid_ch))
            layers.append(nn.ReLU())

        # Depthwise - no BN/activation per MetaNet spec
        layers.append(
            nn.Conv2d(
                mid_ch,
                mid_ch,
                kernel_size,
                stride,
                padding,
                groups=dw_groups,
                bias=False,
            )
        )

        # Projection
        layers.append(nn.Conv2d(mid_ch, out_ch, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


def _make_ir_blocks(in_ch, out_ch, n, expand_ratio, kernel_size=3, group_size=None):
    blocks = []
    for i in range(n):
        c_in = in_ch if i == 0 else out_ch
        blocks.append(
            InvertedResidual(
                c_in,
                out_ch,
                stride=1,
                expand_ratio=expand_ratio,
                kernel_size=kernel_size,
                group_size=group_size,
            )
        )
    return blocks


class MetaNetCNN_Large_HTP(nn.Module):
    """MobileNetV2-style backbone matching MetaNetCNN_Large_HTP.

    Output: [B, 288, 16, 16] for 512x512 input.
    """

    def __init__(self):
        super().__init__()

        # Stage 0: Conv3x3(3->32, s=2) + 3x IR_group(32->32, e=2)
        self.stage0 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            *_make_ir_blocks(32, 32, 3, expand_ratio=2, group_size=32),
        )

        # Stage 1: IR5_group(32->32, e=4, s=2) + 4x IR_group(32->32, e=2)
        self.stage1 = nn.Sequential(
            InvertedResidual(
                32, 32, stride=2, expand_ratio=4, kernel_size=5, group_size=32
            ),
            *_make_ir_blocks(32, 32, 4, expand_ratio=2, group_size=32),
        )

        # Stage 2: IR5(32->64, e=4, s=2) + 4x IR(64->64, e=3)
        self.stage2 = nn.Sequential(
            InvertedResidual(32, 64, stride=2, expand_ratio=4, kernel_size=5),
            *_make_ir_blocks(64, 64, 4, expand_ratio=3),
        )

        # Stage 3: IR5(64->96, e=5, s=2) + 4x IR(96->96, e=3)
        #         + IR(96->160, e=5) + 8x IR(160->160, e=3)
        self.stage3 = nn.Sequential(
            InvertedResidual(64, 96, stride=2, expand_ratio=5, kernel_size=5),
            *_make_ir_blocks(96, 96, 4, expand_ratio=3),
            InvertedResidual(96, 160, stride=1, expand_ratio=5),
            *_make_ir_blocks(160, 160, 8, expand_ratio=3),
        )

        # Stage 4: IR(160->256, e=6, s=2) + 6x IR(256->256, e=5)
        #         + IR(256->288, e=6)
        self.stage4 = nn.Sequential(
            InvertedResidual(160, 256, stride=2, expand_ratio=6),
            *_make_ir_blocks(256, 256, 6, expand_ratio=5),
            InvertedResidual(256, 288, stride=1, expand_ratio=6),
        )

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


class EncoderCNN(nn.Module):
    """Wraps MetaNet backbone with RGB normalization, global pooling, and
    projection to 1024-dim features."""

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self):
        super().__init__()
        self.backbone = MetaNetCNN_Large_HTP()
        self.projection = nn.Linear(576, 1024)
        self.bn = nn.BatchNorm1d(1024)

        self.register_buffer(
            "mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1)
        )

    def forward(self, x):
        # RGB normalization
        x = (x - self.mean) / self.std

        # Backbone -> [B, 288, H/32, W/32]
        x = self.backbone(x)

        # Flatten spatial dims -> [B, 288, H/32 * W/32]
        x = torch.flatten(x, 2)

        # Global pooling: max + mean over spatial dim
        x_max = torch.amax(x, dim=-1)  # [B, 288]
        x_mean = torch.mean(x, dim=-1)  # [B, 288]

        # Concatenate -> [B, 576]
        x = torch.cat([x_max, x_mean], dim=1)

        # Project -> [B, 1024]
        x = self.projection(x)
        x = self.bn(x)

        return x


class MultiLabelPredictionHead(nn.Module):
    """Stack of Linear -> BN1d -> GELU layers, followed by a final Linear."""

    def __init__(self, in_features, hidden_dims, out_features):
        super().__init__()
        layers = []
        prev_dim = in_features
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.GELU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, out_features))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class cnnToy(nn.Module):
    """Toy Conv Model: shared CNN backbone with 9 prediction heads."""

    def __init__(self):
        super().__init__()
        self.encoder = EncoderCNN()

        # 3-hidden-layer heads
        self.dit_topic_head = MultiLabelPredictionHead(1024, [512, 512, 512], N_TOPICS)
        self.dit_subtopic_head = MultiLabelPredictionHead(
            1024, [512, 512, 512], N_SUBTOPICS
        )
        self.concept_head = MultiLabelPredictionHead(1024, [512, 512, 512], N_CONCEPTS)
        self.keyword_head = MultiLabelPredictionHead(
            1024, [512, 512, 512], N_KEYWORDS
        )

        # 4-hidden-layer heads
        self.aesthetics_head = MultiLabelPredictionHead(
            1024, [512, 512, 512, 512], 9
        )
        self.distasteful_head = MultiLabelPredictionHead(
            1024, [512, 512, 512, 512], N_DISTASTEFUL
        )
        self.integrity_ansa_head = MultiLabelPredictionHead(
            1024, [512, 512, 512, 512], 1
        )
        self.integrity_gv_head = MultiLabelPredictionHead(
            1024, [512, 512, 512, 512], 1
        )
        self.vibes_head = MultiLabelPredictionHead(1024, [512, 512, 512, 512], 22)

    def forward(self, x):
        features = self.encoder(x)  # [B, 1024]

        dit_topic = torch.sigmoid(self.dit_topic_head(features))
        dit_subtopic = torch.sigmoid(self.dit_subtopic_head(features))
        concept = torch.sigmoid(self.concept_head(features))
        keyword = torch.sigmoid(self.keyword_head(features))
        aesthetics = self.aesthetics_head(features)  # raw, no sigmoid
        distasteful = torch.sigmoid(self.distasteful_head(features))
        ansa = torch.sigmoid(self.integrity_ansa_head(features))
        gv = torch.sigmoid(self.integrity_gv_head(features))
        vibes = torch.sigmoid(self.vibes_head(features))
        image_features = F.normalize(features, p=2.0, dim=1)

        return (
            dit_topic,
            dit_subtopic,
            concept,
            keyword,
            aesthetics,
            distasteful,
            ansa,
            gv,
            vibes,
            image_features,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Export toy Conv Model model to ExecuTorch Vulkan backend"
    )
    parser.add_argument("-o", "--output_dir", default=".", help="Output directory")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Force fp16 precision inside the Vulkan delegate",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(42)
    model = cnnToy()
    model.eval()

    sample_inputs = (torch.randn(1, 3, 512, 512),)

    logger.info("Exporting model with torch.export...")
    program = export(model, sample_inputs, strict=True)

    compile_options = {}
    if args.fp16:
        compile_options["force_fp16"] = True

    logger.info("Lowering to Vulkan backend...")
    edge_program = to_edge_transform_and_lower(
        program,
        partitioner=[VulkanPartitioner(compile_options)],
    )

    logger.info(f"Lowered graph:\n{edge_program.exported_program().graph}")

    exec_prog = edge_program.to_executorch()

    suffix = "fp16" if args.fp16 else "fp32"
    filename = f"cnn_toy_vulkan_{suffix}"
    save_pte_program(exec_prog, filename, args.output_dir)
    logger.info(f"Saved {filename}.pte to {args.output_dir}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
