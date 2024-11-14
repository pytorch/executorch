# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the same directory.

# Source: https://github.com/yformer/EfficientSAM/blob/main/efficient_sam/build_efficient_sam.py

from .efficient_sam import build_efficient_sam


def build_efficient_sam_vitt():
    return build_efficient_sam(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        checkpoint="https://huggingface.co/merve/EfficientSAM/resolve/main/efficient_sam_vitt.pt",
    ).eval()


def build_efficient_sam_vits():
    return build_efficient_sam(
        encoder_patch_embed_dim=384,
        encoder_num_heads=6,
        checkpoint="weights/efficient_sam_vits.pt",
    ).eval()
