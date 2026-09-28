# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

from executorch.examples.models.mlperf_tiny import MobileNetV1025
from executorch.examples.nxp.models.mlperf_tiny.mlperf_tiny_model import MLPerfTinyModel

log = logging.getLogger(__name__)


class MLPerfTinyVisualWakeWords(MLPerfTinyModel):
    """MLPerf Tiny visual wake words model (MobileNetV1 width 0.25)."""

    # MobileNetV1 specific QAT training hyperparameters.
    TRAIN_HYPERPARAMETERS = {
        "num_epochs": 15,
        "batch_size": 20,
        "lr": 2.5e-7,
        "eps": 1e-7,
        "weight_decay": 0.0,
    }

    INPUT_SHAPE = (1, 3, 96, 96)
    IDX_TO_LABEL = {0: "person", 1: "non_person"}

    # MobileNetV1 stacks 13 depthwise-separable blocks, each with a BatchNorm.
    # Randomly initialized MobileNetV1 has the BatchNorm nodes with default parameters
    # (running_mean=0, running_var=1), causing each depthwise-separable block to behave as identity.
    # In other MLPerf Tiny models, this could be fixed by multiplying the random weights
    # by constant, however in this case the model is too deep and that solution
    # is no longer viable.
    # Instead calibration of BatchNorm by running few forward passes in train mode fixes it.
    BN_CALIBRATION_ITERS = 10
    BN_CALIBRATION_BATCH_SIZE = 16

    def _calibrate_batch_norm(self, model: torch.nn.Module) -> None:
        model.train()
        with torch.no_grad():
            for _ in range(self.BN_CALIBRATION_ITERS):
                inputs = torch.rand(
                    (self.BN_CALIBRATION_BATCH_SIZE, *self.input_shape[1:]),
                    dtype=torch.float32,
                )
                model(inputs)
        model.eval()

    @property
    def input_shape(self):
        return self.INPUT_SHAPE

    @property
    def labels(self):
        return self.IDX_TO_LABEL

    def _init_eager_model(self) -> torch.nn.Module:
        model = MobileNetV1025()
        self._calibrate_batch_norm(model)

        return model.eval()
