# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
from abc import abstractmethod
from pathlib import Path
from typing import Iterator

import torch
from executorch.backends.nxp.tests.calibration_dataset import (
    CalibrationDataset,
    RandomCalibrationDataset,
)
from executorch.examples.models import model_base
from torch.utils.data import DataLoader, Dataset
from torchao.quantization.pt2e import disable_observer
from tqdm import tqdm

log = logging.getLogger(__name__)


class MLPerfTinyModel(model_base.EagerModelBase):
    """Base class of the MLPerf Tiny models."""

    # Default QAT training hyperparameters. Subclasses may override them.
    TRAIN_HYPERPARAMETERS = {
        "num_epochs": 15,
        "batch_size": 64,
        "lr": 5e-6,
        "eps": 1e-7,
        "weight_decay": 1e-4,
    }

    def __init__(
        self,
        dataset_path: Path | str | None = None,
        use_random_dataset: bool = False,
        num_samples: int | None = None,
        num_workers: int = 4,
    ):
        """
        Create the model wrapper along with the dataset it owns.
        If `use_random_dataset = True`, then `num_samples` must be set.
        If `use_random_dataset = False`, then `dataset_path` must be set.
        """
        self._num_workers = num_workers
        self._num_samples = num_samples
        self._use_random_dataset = use_random_dataset
        self._dataset_path = dataset_path

        # throws ValueError if validation fails
        self._validate_args()

        self._eager_model = self._init_eager_model()
        self.dataset = self._init_dataset()

    @staticmethod
    def _collate_fn(data: list[tuple]):
        data, labels = zip(*data)
        return torch.stack(list(data)), torch.tensor(list(labels))

    @abstractmethod
    def _init_eager_model(self) -> torch.nn.Module:
        pass

    @property
    @abstractmethod
    def input_shape(self):
        pass

    @property
    @abstractmethod
    def labels(self):
        pass

    def _validate_args(self):
        valid_num_samples = isinstance(self._num_samples, int) and self._num_samples > 0

        if self._use_random_dataset:
            if not valid_num_samples:
                raise ValueError(
                    f"Invalid number of samples to randomly generate. Got {self._num_samples}."
                )

        else:
            if valid_num_samples:
                raise ValueError(
                    "Num samples was supplied, but it is omitted because `use_random_dataset=False`."
                )

            if self._dataset_path is None or not os.path.exists(self._dataset_path):
                raise ValueError(
                    f"Invalid dataset path for loading the data. Got {self._dataset_path}."
                )

    def get_qat_train_inputs(
        self, batch_size: int = 5, dataset_portion: float = 0.1
    ) -> Iterator[tuple[torch.Tensor]]:
        """Return an iterator over a portion of the model dataset, to be used for QAT."""
        reduced_dataset = torch.utils.data.Subset(
            self.dataset, range(int(len(self.dataset) * dataset_portion))
        )
        reduced_loader = DataLoader(
            reduced_dataset,
            batch_size=batch_size,
            collate_fn=self._collate_fn,
            num_workers=self._num_workers,
            pin_memory=True,
        )
        return iter(reduced_loader)

    def get_calibration_inputs(
        self, batch_size: int = 1
    ) -> Iterator[tuple[torch.Tensor]]:
        """Return an iterator over the model dataset, to be used for post training quantization."""
        loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            collate_fn=self._collate_fn,
            num_workers=self._num_workers,
            pin_memory=True,
        )
        return itertools.starmap(lambda data, _: (data,), iter(loader))

    def get_eager_model(self):
        return self._eager_model

    def get_example_inputs(self) -> tuple[torch.Tensor]:
        return (torch.randn(self.input_shape, dtype=torch.float32),)

    def train_model_fn(
        self, model, num_epochs=None, batch_size=None, channels_last=False
    ):
        hyperparameters = self.TRAIN_HYPERPARAMETERS
        num_epochs = (
            num_epochs if num_epochs is not None else hyperparameters["num_epochs"]
        )
        batch_size = (
            batch_size if batch_size is not None else hyperparameters["batch_size"]
        )

        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)

        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=hyperparameters["lr"],
            eps=hyperparameters["eps"],
            weight_decay=hyperparameters["weight_decay"],
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        log.warning("Starting training...")

        data = self.get_qat_train_inputs(batch_size=batch_size)
        for nepoch in range(num_epochs):
            for samples, labels in tqdm(data):
                if channels_last:
                    samples = samples.to(memory_format=torch.channels_last)

                optimizer.zero_grad()
                outputs = model(samples)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

            if nepoch >= num_epochs / 3:
                model.apply(disable_observer)

        return model

    def _init_dataset(self) -> Dataset:
        if self._use_random_dataset:
            num_classes = len(self.labels)
            sample_shape = tuple(self.input_shape)[1:]
            return RandomCalibrationDataset(
                self._num_samples, sample_shape, num_classes
            )
        else:
            return CalibrationDataset(self._dataset_path)
