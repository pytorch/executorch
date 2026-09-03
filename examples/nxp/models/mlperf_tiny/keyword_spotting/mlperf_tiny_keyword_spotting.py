# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path

import torch

from executorch.backends.nxp.tests.calibration_dataset import (
    CalibrationDataset,
    RandomCalibrationDataset,
)

from executorch.examples.models.mlperf_tiny import DSCNNKWS
from executorch.examples.nxp.models.mlperf_tiny.mlperf_tiny_model import MLPerfTinyModel
from torch.utils.data import Dataset


log = logging.getLogger(__name__)

INPUT_SHAPE = (1, 1, 49, 10)
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


class MLPerfTinyKeywordSpotting(MLPerfTinyModel):
    """MLPerf Tiny keyword spotting model (DS-CNN)."""

    def __init__(
        self,
        num_samples: int = 200,
        dataset_path: Path | str | None = None,
        use_random_dataset: bool = False,
    ):
        self._num_samples = num_samples
        self._use_random_dataset = use_random_dataset
        self._dataset_path = dataset_path

        super().__init__()

    @property
    def input_shape(self):
        return INPUT_SHAPE

    @property
    def labels(self):
        return IDX_TO_LABEL

    def _init_dataset(self) -> Dataset:
        if self._use_random_dataset:
            num_classes = len(self.labels)
            sample_shape = tuple(self.input_shape)[1:]
            return RandomCalibrationDataset(
                self._num_samples, sample_shape, num_classes
            )
        else:
            if self._dataset_path is None:
                raise ValueError(
                    "Path to dataset data cannot be empty. If you want to use random data, set `use_random_dataset = True`"
                )
            return CalibrationDataset(self._dataset_path)

    def _init_eager_model(self) -> torch.nn.Module:
        num_classes = len(self.labels)
        return DSCNNKWS(num_classes)
