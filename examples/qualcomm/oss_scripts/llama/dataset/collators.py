# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from executorch.examples.qualcomm.oss_scripts.llama.masking_utils import AttentionMask


def _pad_and_stack(
    token_ids: List[List[int]],
    max_len: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.stack(
        [
            F.pad(torch.tensor(seq, dtype=dtype), (0, max(0, max_len - len(seq))))
            for seq in token_ids
        ]
    )


class ModalityEncoderCollator:
    """Collator for encoder batches (audio or vision).

    Each item from ModalityEncoderDataset is a Tuple[Tensor, ...] (one set of
    encoder inputs per media file).

    Returns:
        {"inputs": torch.Tensor}
    """

    def __call__(self, batch: List[Tuple[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {"inputs": torch.stack([item[0] for item in batch])}


class LLMCalibCollator:
    """Collator for PTQ calibration batches — pads tokens and builds attention masks.

    Returns:
        {
            "input_ids": Tensor,              # padded + stacked, (batch_size, max_len)
            "token_ids": List[List[int]],     # original, unpadded token_ids
            "attention_mask": Tuple[Tensor, ...],
        }
    """

    def __init__(
        self,
        attn_mask_template: AttentionMask,
        max_context_len: int,
        token_dtype: torch.dtype = torch.int32,
    ) -> None:
        self._template = attn_mask_template
        self._max_context_len = max_context_len
        self._token_dtype = token_dtype

    def __call__(self, batch: List[Tuple]) -> Dict[str, Union[torch.Tensor, Tuple]]:
        token_ids = [item[0] for item in batch]
        attn = AttentionMask.from_input_ids(
            self._template, token_ids, self._max_context_len
        )
        attention_mask = tuple(m.mask for m in attn.masks)
        return {
            "input_ids": _pad_and_stack(
                token_ids, self._max_context_len, self._token_dtype
            ),
            "attention_mask": attention_mask,
            "labels": None,
            "token_ids": token_ids,
        }
