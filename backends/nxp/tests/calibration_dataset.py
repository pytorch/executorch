# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import lzma
import pickle

import torch

from torch.utils.data.dataset import Dataset


class CalibrationDataset(Dataset):
    def __init__(self, data_path):
        if data_path.endswith(".xz"):
            with lzma.open(data_path) as f:
                self.examples = pickle.load(f)
        elif data_path.endswith(".pt"):
            self.examples = torch.load(
                data_path, map_location=torch.device("cpu"), weights_only=False
            )
        else:
            raise ValueError("Invalid file format, supported formats are .xz, .pt.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


class RandomCalibrationDataset(Dataset):
    def __init__(
        self,
        num_examples: int,
        sample_shape,
        num_classes: int,
        balanced: bool = True,
    ):
        if balanced and num_examples % num_classes != 0:
            raise ValueError(
                f"RandomCalibrationDataset: `num_examples` ({num_examples}) must be divisible by "
                f"`num_classes` ({num_classes}) when `balanced` is True."
            )

        self._num_examples = num_examples
        self._shape = tuple(sample_shape)
        self._num_classes = num_classes
        self._balanced = balanced

        labels = self._create_labels(num_examples, num_classes, balanced)

        self.examples = []
        for label in labels:
            data = torch.rand(self._shape, dtype=torch.float32)
            self.examples.append((data, label))

    @staticmethod
    def _create_labels(
        num_examples: int, num_classes: int, balanced: bool
    ) -> list[int]:
        """Create the list of labels for the individual examples."""
        if balanced:
            examples_per_class = num_examples // num_classes
            labels = torch.arange(num_classes).repeat_interleave(examples_per_class)
            labels = labels[torch.randperm(num_examples)]
        else:
            labels = torch.randint(low=0, high=num_classes, size=(num_examples,))

        return labels.tolist()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]
