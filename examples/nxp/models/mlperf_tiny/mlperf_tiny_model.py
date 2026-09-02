# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from abc import abstractmethod
from typing import Iterator

import torch
from executorch.examples.models import model_base
from torch.utils.data import DataLoader, Dataset


class MLPerfTinyModel(model_base.EagerModelBase):
    """Base class of the MLPerf Tiny models."""

    def __init__(self):
        """Create the model wrapper along with the dataset it owns."""
        self._num_workers = 4

        self._eager_model = self._init_eager_model()
        self.dataset = self._init_dataset()

    @staticmethod
    def _collate_fn(data: list[tuple]):
        data, labels = zip(*data)
        return torch.stack(list(data)), torch.tensor(list(labels))

    @abstractmethod
    def _init_dataset(self) -> Dataset:
        pass

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
