# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

from torch.utils.data import Dataset


class LLMDataset(Dataset):
    """TEXT_DECODER dataset — pure token storage, no padding or masking.

    sequences: raw (unpadded) token-id lists, one per sample.
    """

    def __init__(
        self,
        sequences: List[List[int]],
    ):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[List[int]]:
        return (self.sequences[idx],)


class ModalityEncoderDataset(Dataset):
    """Encoder calibration data for a single modality."""

    def __init__(self, data: List[List]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
