# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import lzma
import pickle

import numpy as np
import torch

from torch.utils.data.dataset import Dataset


class CalibrationDataset(Dataset):
    def __init__(self, data_path: str) -> None:
        if data_path.endswith(".xz"):
            with lzma.open(data_path) as f:
                self.examples = pickle.load(f)
        elif data_path.endswith(".pt"):
            self.examples = torch.load(
                data_path, map_location=torch.device("cpu"), weights_only=False
            )
        else:
            raise ValueError("Invalid file format, supported formats are .xz, .pt.")

        # Ensure all data items are torch.Tensor regardless of source format.
        self.examples = [self._to_tensor_example(ex) for ex in self.examples]

    @staticmethod
    def _to_tensor(data) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return data
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        raise TypeError(f"Expected Tensor or ndarray, got {type(data).__name__}.")

    @classmethod
    def _to_tensor_example(cls, example: tuple) -> tuple:
        if not isinstance(example, tuple):
            raise NotImplementedError("Examples other than tuple are not supported.")
        if len(example) == 2:
            data, label = example
        elif len(example) == 1:
            data = example[0]
            label = None
        else:
            raise ValueError(
                f"Unexpected number of elements in example: {len(example)}."
            )

        if isinstance(data, (torch.Tensor, np.ndarray)):
            return cls._to_tensor(data), label
        if isinstance(data, list):
            return [cls._to_tensor(d) for d in data], label
        raise TypeError(f"Unsupported data type in example: {type(data).__name__}.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> tuple:
        return self.examples[i]


class RandomCalibrationDataset(Dataset):
    def __init__(
        self,
        num_examples: int,
        sample_shape: tuple[int, ...] | list[int],
        num_classes: int,
        balanced: bool = True,
    ) -> None:
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

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        return self.examples[i]
