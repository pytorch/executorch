# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

from executorch.examples.models.mlperf_tiny import DSCNNKWS
from executorch.examples.nxp.models.mlperf_tiny.mlperf_tiny_model import MLPerfTinyModel

log = logging.getLogger(__name__)


class MLPerfTinyKeywordSpotting(MLPerfTinyModel):
    """MLPerf Tiny keyword spotting model (DS-CNN)."""

    INPUT_SHAPE = (1, 1, 49, 10)
    # Because of the architecture of the model,
    # non-scaled random weights tend to produce zero tensors,
    # making it hard to compute numerical accuracy of the delegated model.
    # Scaling the random weights makes the model produce reasonable results.
    WEIGHT_INIT_SCALE = 2.0

    IDX_TO_LABEL = {
        0: "Down",
        1: "Go",
        2: "Left",
        3: "No",
        4: "Off",
        5: "On",
        6: "Right",
        7: "Stop",
        8: "Up",
        9: "Yes",
        10: "Silence",
        11: "Unknown",
    }

    @property
    def input_shape(self):
        return self.INPUT_SHAPE

    @property
    def labels(self):
        return self.IDX_TO_LABEL

    def _init_weights(self, model: torch.nn.Module):
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    module.weight *= self.WEIGHT_INIT_SCALE

    def _init_eager_model(self) -> torch.nn.Module:
        num_classes = len(self.labels)
        model = DSCNNKWS(num_classes)
        self._init_weights(model)

        return model.eval()
