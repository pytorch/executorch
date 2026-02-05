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
        if data_path.endswith('.xz'):
            with lzma.open(data_path) as f:
                self.examples = pickle.load(f)
        elif data_path.endswith('.pt'):
            self.examples = torch.load(data_path, map_location=torch.device('cpu'), weights_only=False)
        else:
            raise ValueError('Invalid file format, supported formats are .xz, .pt.')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]
