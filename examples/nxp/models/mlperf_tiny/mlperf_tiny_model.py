# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import Iterator

import torch
from executorch.backends.nxp.tests.calibration_dataset import CalibrationDataset
from executorch.examples.models import model_base
from torch.utils.data import DataLoader, Dataset

from executorch.examples.nxp.models.model_manager import ModelManager


class RandomTensorDataset(Dataset):
    def __init__(self, sample_shape: tuple[int, int], num_samples: int, num_classes: int):
        self._sample_shape = sample_shape
        self._num_samples = num_samples
        self._num_classes = num_classes

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, index: int):
        if index < 0 or index >= self._num_samples:
            raise IndexError(index)
        data = torch.randn(self._sample_shape, dtype=torch.float32)
        label = torch.randint(0, self._num_classes, (1,)).item()
        return data, label


class MLPerfTinyModel(model_base.EagerModelBase):
    def __init__(self):
        self._batch_size = 1
        self._dataset = None
        self._model_manager = ModelManager()
        self._num_workers = 4
        self._num_samples = 128

    @property
    @abstractmethod
    def _input_shape(self):
        pass

    @staticmethod
    def _collate_fn(data: list[tuple]):
        data, labels = zip(*data)
        return torch.stack(list(data)), torch.tensor(list(labels))

    def get_qat_train_inputs(self, batch_size: int = 5, dataset_portion: float = 0.1) -> Iterator[tuple[torch.Tensor]]:
        data_loader = self._get_data_loader()
        reduced_dataset = torch.utils.data.Subset(
            data_loader.dataset,
            range(int(len(data_loader.dataset) * dataset_portion))
        )
        reduced_loader = DataLoader(
            reduced_dataset,
            batch_size=batch_size,
            collate_fn=self._collate_fn,
            num_workers=self._num_workers,
            pin_memory=True
        )
        return iter(reduced_loader)
  
    def get_example_inputs(self) -> tuple[torch.Tensor]:
        return (torch.randn(self._input_shape, dtype=torch.float32),)

    def _get_data_loader(self):
        self._init_dataset()
        data_loader = DataLoader(self._dataset, batch_size=self._batch_size,
                                 collate_fn=self._collate_fn,
                                 num_workers=self._num_workers, pin_memory=True)
        return data_loader

    def _init_dataset(self):
        if self._dataset is None:
            sample_shape = tuple(self._input_shape)[1:]
            self._dataset = RandomTensorDataset(sample_shape, num_samples=self._num_samples)
