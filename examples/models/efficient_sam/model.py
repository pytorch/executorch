# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

from ..model_base import EagerModelBase

from .efficient_sam_core.build_efficient_sam import build_efficient_sam_vitt


class EfficientSAM(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("Loading EfficientSAM model")
        efficient_sam = build_efficient_sam_vitt()
        logging.info("Loaded EfficientSAM model")
        return efficient_sam

    def get_example_inputs(self):
        B, H, W = 1, 1024, 1024
        num_queries, num_pts = 1, 1

        batched_images = torch.randn((B, 3, H, W))
        batched_points = torch.rand((B, num_queries, num_pts, 2)) * torch.tensor([H, W])
        batched_point_labels = torch.ones((B, num_queries, num_pts))

        return (batched_images, batched_points, batched_point_labels)
