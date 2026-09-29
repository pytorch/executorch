# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from executorch.examples.qualcomm.oss_scripts.llama.dataset.targets import (
    make_causal_labels,
    make_conversation_labels,
)
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


class LLMTrainingCollator(LLMCalibCollator):
    """Collator for QAT training batches — pads tokens, builds masks, and appends labels.

    When assistant_masks are present (from HF chat datasets), only assistant-turn
    positions are supervised. Otherwise causal next-token labels are used.
    """

    _warned_empty_assistant_mask = False

    def __call__(
        self, batch: List[Tuple]
    ) -> Dict[str, Union[torch.Tensor, Tuple, List]]:
        model_inputs = super().__call__(batch)
        assistant_masks = [item[1] for item in batch]

        model_inputs["labels"] = torch.stack(
            [
                torch.tensor(
                    self._make_labels(seq, mask),
                    dtype=torch.int64,
                )
                for seq, mask in zip(model_inputs["token_ids"], assistant_masks)
            ]
        )

        return model_inputs

    def _make_labels(self, tokens: List[int], assistant_mask: List[int]) -> List[int]:
        """Assistant-only labels, warning once when a mask supervises nothing.

        A None or all-zero mask means the sample has no assistant turns, so
        make_conversation_labels falls back to full causal next-token labels.
        """
        if assistant_mask is None or not any(assistant_mask):
            if not LLMTrainingCollator._warned_empty_assistant_mask:
                logging.warning(
                    "assistant_mask is empty or all zeros (plain-text corpus, or "
                    "chat template missing `{%% generation %%}` markers); falling "
                    "back to causal next-token labels. Supervision will cover the "
                    "full sequence, not just assistant turns."
                )
                LLMTrainingCollator._warned_empty_assistant_mask = True
            return make_causal_labels(tokens, self._max_context_len)

        return make_conversation_labels(tokens, assistant_mask, self._max_context_len)
