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

from executorch.examples.models.mlperf_tiny import ResNet8
from executorch.examples.nxp.models.mlperf_tiny.mlperf_tiny_model import MLPerfTinyModel
from torch.utils.data import Dataset
from torchao.quantization.pt2e import disable_observer
from tqdm import tqdm

log = logging.getLogger(__name__)

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


class MLPerfTinyImageClassification(MLPerfTinyModel):
    """MLPerf Tiny image classification model (ResNet-8)."""

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
        return ResNet8(num_classes)

    def train_model_fn(self, model, num_epochs=15, batch_size=20, channels_last=False):
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)

        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=1e-5,
            weight_decay=1e-4,
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        logging.warning("Starting training...")

        data = self.get_qat_train_inputs(batch_size=batch_size)
        for nepoch in range(num_epochs):
            for images, labels in tqdm(data):
                if channels_last:
                    images = images.to(memory_format=torch.channels_last)

                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

            if nepoch >= num_epochs / 3:
                model.apply(disable_observer)

        return model
