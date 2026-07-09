# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import torch

PADDING_MASK_VALUE = -255.0


def create_causal_attn_mask(max_batch_size: int, ar_len: int, max_context_len: int):
    """
    Creating a causal attention mask (ar_len: 5, max_context_len: 15)
        0 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ○ ○ ○ ○
        1 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ○ ○ ○
        2 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ● ○ ○
        3 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ● ● ○
        4 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ● ● ●

    ● = activate (can attend), ○ = inactivate (masked)
    """
    mask = torch.full((ar_len, ar_len), PADDING_MASK_VALUE)
    mask_cond = torch.arange(ar_len)
    mask.masked_fill_(mask_cond.view(1, ar_len) <= mask_cond.view(ar_len, 1), 0)

    if max_context_len != ar_len:
        mask = torch.cat(
            [
                torch.ones(ar_len, max_context_len - ar_len) * PADDING_MASK_VALUE,
                mask,
            ],
            dim=-1,
        )
    mask = mask[None, :, :].expand(max_batch_size, ar_len, max_context_len)
    return mask


def create_sliding_window_attn_mask(
    max_batch_size: int, ar_len: int, max_context_len: int, sliding_window: int
):
    """
    Creating a sliding_window attention mask (ar_len: 5, max_context_len: 15, sliding_window: 3)
        0 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ○ ○ ○ ○
        1 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ○ ○ ○
        2 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ● ○ ○
        3 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ● ○
        4 ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ●

    ● = activate (can attend), ○ = inactivate (masked)
    """
    mask = torch.full((ar_len, ar_len), PADDING_MASK_VALUE)
    mask_cond = torch.arange(ar_len)
    mask.masked_fill_(
        (mask_cond.view(1, ar_len) <= mask_cond.view(ar_len, 1))
        & (mask_cond.view(ar_len, 1) - mask_cond.view(1, ar_len) < sliding_window),
        0,
    )

    if max_context_len != ar_len:
        mask = torch.cat(
            [
                torch.ones(ar_len, max_context_len - ar_len) * PADDING_MASK_VALUE,
                mask,
            ],
            dim=-1,
        )
    mask = mask[None, :, :].expand(max_batch_size, ar_len, max_context_len)
    return mask


class BaseAttentionMask(ABC):
    def __init__(self, max_batch_size: int, ar_len: int, max_context_len: int):
        """
        Base class for attention masks used in autoregressive or hybrid attention mechanisms.

        Args:
            max_batch_size (int): Maximum batch size supported.
            ar_len (int): Length of the autoregressive sequence.
            max_context_len (int): Maximum sequence length.
        """
        self.max_batch_size = max_batch_size
        self.ar_len = ar_len
        self.max_context_len = max_context_len

    @property
    @abstractmethod
    def mask(self) -> torch.Tensor:
        """
        Attention mask tensor that must be initialized by child classes.
        """
        pass

    @abstractmethod
    def smart_mask_init(self, pos):
        """
        Initialize the attention mask by smart mask initialization method after model forward.
        Args:
            pos (int): Current position in the sequence.
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

    def _extra_init_kwargs(self) -> dict:
        return {}

    def _mask_padding_positions(
        self, input_ids: List[List[int]], max_seq_length: int
    ) -> None:
        """Mask positions beyond each sequence's actual length."""
        actual_lens = torch.tensor([len(seq) for seq in input_ids])
        pad_rows = torch.arange(max_seq_length).unsqueeze(0) >= actual_lens.unsqueeze(1)
        self.mask.masked_fill_(pad_rows.unsqueeze(-1), PADDING_MASK_VALUE)


class CausalAttentionMask(BaseAttentionMask):
    def __init__(self, max_batch_size: int, ar_len: int, max_context_len: int):
        super().__init__(max_batch_size, ar_len, max_context_len)
        self._max_batch_size = max_batch_size
        self._mask = create_causal_attn_mask(max_batch_size, ar_len, max_context_len)

    @property
    def mask(self):
        return self._mask

    def smart_mask_init(self, pos):
        self._mask = create_causal_attn_mask(
            self.max_batch_size, self.ar_len, self.max_context_len
        )
        self.mask[:, :, :pos] = 0

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

    @classmethod
    def from_input_ids(
        cls, input_ids: List[List[int]], max_seq_length: int, **kwargs
    ) -> "CausalAttentionMask":
        """Build a causal mask and apply padding for variable-length sequences."""
        mask = cls(len(input_ids), max_seq_length, max_seq_length)
        mask._mask = mask._mask.clone()
        mask._mask_padding_positions(input_ids, max_seq_length)
        return mask


class SlidingWindowAttentionMask(BaseAttentionMask):
    def __init__(
        self,
        max_batch_size: int,
        ar_len: int,
        max_context_len: int,
        sliding_window: int,
    ):
        super().__init__(max_batch_size, ar_len, max_context_len)
        self._mask = create_sliding_window_attn_mask(
            max_batch_size, ar_len, max_context_len, sliding_window
        )
        self.sliding_window = sliding_window

    @property
    def mask(self):
        return self._mask

    def smart_mask_init(self, pos):
        self._mask = create_sliding_window_attn_mask(
            self.max_batch_size, self.ar_len, self.max_context_len, self.sliding_window
        )
        self.mask[:, :, :pos] = 0

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
            Sliding window shifts again, masking older positions and activate new position.
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
                self.mask[:, i, : end_pos - available_cache_len] = PADDING_MASK_VALUE

    def _extra_init_kwargs(self) -> dict:
        return {"sliding_window": self.sliding_window}

    @classmethod
    def from_input_ids(
        cls,
        input_ids: List[List[int]],
        max_seq_length: int,
        sliding_window: int,
        **kwargs,
    ) -> "SlidingWindowAttentionMask":
        """Build a sliding-window mask and apply padding for variable-length sequences."""
        mask = cls(len(input_ids), max_seq_length, max_seq_length, sliding_window)
        mask._mask = mask._mask.clone()
        mask._mask_padding_positions(input_ids, max_seq_length)
        return mask


class AttentionMask:
    def __init__(self, masks: Union[BaseAttentionMask, List[BaseAttentionMask]]):
        self.masks = masks if isinstance(masks, list) else [masks]

    def smart_mask_init(self, pos):
        for mask in self.masks:
            mask.smart_mask_init(pos)

    def smart_mask_update(self, pos, n_updates, lade_pos_offset=None):
        for mask in self.masks:
            mask.smart_mask_update(pos, n_updates, lade_pos_offset)

    def __iter__(self):
        return iter([mask.mask for mask in self.masks])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return tuple(m.mask[idx] for m in self.masks)

    @classmethod
    def from_input_ids(
        cls,
        template: "AttentionMask",
        input_ids: List[List[int]],
        max_seq_length: int,
    ) -> "AttentionMask":
        """
        Build a calibration AttentionMask that mirrors template's mask types.

        Delegates construction to each mask's own classmethod so that adding a
        new mask type only requires implementing from_input_ids on that class —
        no edits needed here.
        """
        masks = [
            type(base_mask).from_input_ids(
                input_ids, max_seq_length, **base_mask._extra_init_kwargs()
            )
            for base_mask in template.masks
        ]
        return cls(masks)
