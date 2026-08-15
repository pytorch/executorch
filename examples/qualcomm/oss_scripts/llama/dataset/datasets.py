# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

from torch.utils.data import Dataset


class LLMDataset(Dataset):
    """TEXT_DECODER dataset — pure token storage, no padding or masking.

    sequences:          raw (unpadded) token-id lists, one per sample.
    assistant_masks:    parallel token-level 0/1 mask (1 = assistant turn);
                        None for plain-text tasks (causal labels used instead).
    """

    def __init__(
        self,
        sequences: List[List[int]],
        assistant_masks: Optional[List[List[int]]] = None,
    ):
        self.sequences = sequences
        self.assistant_masks = assistant_masks

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[List[int], Optional[List[int]]]:
        assistant_mask = (
            self.assistant_masks[idx] if self.assistant_masks is not None else None
        )
        return self.sequences[idx], assistant_mask


class ModalityEncoderDataset(Dataset):
    """Encoder calibration data for a single modality."""

    def __init__(self, data: List[List]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
