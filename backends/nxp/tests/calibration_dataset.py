# Copyright 2025 NXP
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
        dtype=torch.float32,
    ):
        self._num_examples = num_examples
        self._shape = tuple(sample_shape)
        self._num_classes = num_classes

        self.examples = []
        for _ in range(num_examples):
            if dtype.is_floating_point:
                data = torch.rand(self._shape, dtype=dtype)
            else:
                data = torch.randint(
                    low=0,
                    high=256,
                    size=self._shape,
                    dtype=dtype,
                )
            label = int(
                torch.randint(
                    low=0, high=num_classes, size=(1,),
                ).item()
            )
            self.examples.append((data, label))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]




