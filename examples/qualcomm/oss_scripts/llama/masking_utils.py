# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import List, Union

import torch


def create_causal_attn_mask(max_batch_size: int, ar_len: int, max_seq_len: int):
    """
    Creating a causal attention mask (ar_len: 5, max_seq_len: 15)
        0 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ○ ○ ○ ○
        1 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ○ ○ ○
        2 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ● ○ ○
        3 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ● ● ○
        4 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ● ● ●

    ● = activate (can attend), ○ = inactivate (masked)
    """
    mask = torch.full((ar_len, ar_len), -255.0)
    mask_cond = torch.arange(ar_len)
    mask.masked_fill_(mask_cond.view(1, ar_len) <= mask_cond.view(ar_len, 1), 0)

    if max_seq_len != ar_len:
        mask = torch.cat(
            [
                torch.ones(ar_len, max_seq_len - ar_len) * -255.0,
                mask,
            ],
            dim=-1,
        )
    mask = mask[None, :, :].expand(max_batch_size, ar_len, max_seq_len)
    return mask


def create_sliding_window_attn_mask(
    max_batch_size: int, ar_len: int, max_seq_len: int, sliding_window: int
):
    """
    Creating a sliding_window attention mask (ar_len: 5, max_seq_len: 15, sliding_window: 3)
        0 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ○ ○ ○ ○
        1 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ○ ○ ○
        2 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ● ○ ○
        3 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ● ○
        4 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ●

    ● = activate (can attend), ○ = inactivate (masked)
    """
    mask = torch.full((ar_len, ar_len), -255.0)
    mask_cond = torch.arange(ar_len)
    mask.masked_fill_(
        (mask_cond.view(1, ar_len) <= mask_cond.view(ar_len, 1))
        & (mask_cond.view(ar_len, 1) - mask_cond.view(1, ar_len) < sliding_window),
        0,
    )

    if max_seq_len != ar_len:
        mask = torch.cat(
            [
                torch.ones(ar_len, max_seq_len - ar_len) * -255.0,
                mask,
            ],
            dim=-1,
        )
    mask = mask[None, :, :].expand(max_batch_size, ar_len, max_seq_len)
    return mask


class BaseAttentionMask(ABC):
    def __init__(self, max_batch_size: int, ar_len: int, max_seq_len: int):
        """
        Base class for attention masks used in autoregressive or hybrid attention mechanisms.

        Args:
            max_batch_size (int): Maximum batch size supported.
            ar_len (int): Length of the autoregressive sequence.
            max_seq_len (int): Maximum sequence length.
        """
        self.max_batch_size = max_batch_size
        self.ar_len = ar_len
        self.max_seq_len = max_seq_len

    @property
    @abstractmethod
    def mask(self) -> torch.Tensor:
        """
        Attention mask tensor that must be initialized by child classes.
        """
        pass

    @abstractmethod
    def smart_mask_update(self, pos, n_updates, lade_pos_offset):
        """
        Update the attention mask by smart mask update method after model forward.

        Args:
            pos (int): Current position in the sequence.
            n_updates (int): Number of new tokens to update.
            lade_pos_offset (List[int]): Position offset of lookahead attention mask.
        """
        pass


class CausalAttentionMask(BaseAttentionMask):
    def __init__(self, max_batch_size: int, ar_len: int, max_seq_len: int):
        super().__init__(max_batch_size, ar_len, max_seq_len)
        self._mask = create_causal_attn_mask(max_batch_size, ar_len, max_seq_len)

    @property
    def mask(self):
        return self._mask

    def smart_mask_update(self, pos, n_updates, _):
        """
        Smart Mask mechanism for attention mask updating

        Initial mask(5x15) layout (before any updates):
            Each row represents a query token in the autoregressive context.
            ● = activate (can attend), ○ = inactivate (masked)

            0 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ○ ○ ○ ○
            1 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ○ ○ ○
            2 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ● ○ ○
            3 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ● ● ○
            4 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ● ● ●

        After 1st update (e.g., pos=0, n_updates=5, sliding_window=3):
            Newly added tokens are unmasked (set to 0).

            0 ● ● ● ● ● ○ ○ ○ ○ ○ ● ○ ○ ○ ○
            1 ● ● ● ● ● ○ ○ ○ ○ ○ ● ● ○ ○ ○
            2 ● ● ● ● ● ○ ○ ○ ○ ○ ● ● ● ○ ○
            3 ● ● ● ● ● ○ ○ ○ ○ ○ ● ● ● ● ○
            4 ● ● ● ● ● ○ ○ ○ ○ ○ ● ● ● ● ●

        After 2nd update (e.g., pos=5, n_updates=5):

            0 ● ● ● ● ● ● ● ● ● ● ● ○ ○ ○ ○
            1 ● ● ● ● ● ● ● ● ● ● ● ● ○ ○ ○
            2 ● ● ● ● ● ● ● ● ● ● ● ● ● ○ ○
            3 ● ● ● ● ● ● ● ● ● ● ● ● ● ● ○
            4 ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●
        """
        start_pos = pos
        end_pos = pos + n_updates
        self.mask[:, :, start_pos:end_pos] = 0


class SlidingWindowAttentionMask(BaseAttentionMask):
    def __init__(
        self,
        max_batch_size: int,
        ar_len: int,
        max_seq_len: int,
        sliding_window: int,
    ):
        super().__init__(max_batch_size, ar_len, max_seq_len)
        self._mask = create_sliding_window_attn_mask(
            max_batch_size, ar_len, max_seq_len, sliding_window
        )
        self.sliding_window = sliding_window

    @property
    def mask(self):
        return self._mask

    def smart_mask_update(self, pos, n_updates, lade_pos_offset):
        """
        Smart Mask mechanism for attention mask updating

        Initial mask(5x15) layout (before any updates):
            Each row represents a query token in the autoregressive context.
            ● = activate (can attend), ○ = inactivate (masked)

            0 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ○ ○ ○ ○
            1 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ○ ○ ○
            2 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ● ○ ○
            3 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ● ○
            4 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ●

        After 1st update (e.g., pos=0, n_updates=5, sliding_window=3):
            Newly added tokens are unmasked (set to 0).
            Earlier tokens lose access to older cache due to sliding window limits.

            0 ○ ○ ○ ● ● ○ ○ ○ ○ ○ ● ○ ○ ○ ○
            1 ○ ○ ○ ○ ● ○ ○ ○ ○ ○ ● ● ○ ○ ○
            2 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ● ○ ○
            3 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ● ○
            4 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ●


        After 2nd update (e.g., pos=5, n_updates=5, sliding_window=3):
            Sliding window shifts again, masking older positions and activate new postion.

            0 ○ ○ ○ ○ ○ ○ ○ ○ ● ● ● ○ ○ ○ ○
            1 ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ● ○ ○ ○
            2 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ● ○ ○
            3 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ● ○
            4 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ●
        """
        start_pos = pos
        end_pos = pos + n_updates
        # Unmask the same range in the sliding window mask
        self.mask[:, :, start_pos:end_pos] = 0

        for i in range(self.ar_len):
            # Calculate how many cached tokens are still available for this row
            available_cache_len = self.sliding_window - (
                (i + 1) if lade_pos_offset is None else (lade_pos_offset[i] + 1)
            )

            # If the current position exceeds available cache, mask the overflow
            if end_pos > available_cache_len:
                # Mask tokens that are no longer within the sliding window
                # TODO: [Optional]: it can be optimized by computing the exact start index
                self.mask[:, i, : end_pos - available_cache_len] = -255.0


class AttentionMask:
    def __init__(self, masks: Union[BaseAttentionMask, List[BaseAttentionMask]]):
        self.masks = masks if isinstance(masks, list) else [masks]

    def smart_mask_update(self, pos, n_updates, lade_pos_offset=None):
        for mask in self.masks:
            mask.smart_mask_update(pos, n_updates, lade_pos_offset)

    def __iter__(self):
        return iter([mask.mask for mask in self.masks])
