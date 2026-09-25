# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

from executorch.examples.models.mlperf_tiny import ResNet8
from executorch.examples.nxp.models.mlperf_tiny.mlperf_tiny_model import MLPerfTinyModel

log = logging.getLogger(__name__)


class MLPerfTinyImageClassification(MLPerfTinyModel):
    """MLPerf Tiny image classification model (ResNet-8)."""

    # ResNet-8 specific QAT training hyperparameters.
    TRAIN_HYPERPARAMETERS = {
        "num_epochs": 15,
        "batch_size": 20,
        "lr": 1e-5,
        "eps": 1e-8,
        "weight_decay": 1e-4,
    }

    INPUT_SHAPE = (1, 3, 32, 32)
    IDX_TO_LABEL = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }

    @property
    def input_shape(self):
        return self.INPUT_SHAPE

    @property
    def labels(self):
        return self.IDX_TO_LABEL

    def _init_eager_model(self) -> torch.nn.Module:
        num_classes = len(self.labels)
        return ResNet8(num_classes).eval()
