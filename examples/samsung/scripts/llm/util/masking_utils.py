# Copyright (c) Qualcomm Innovation Center, Inc.
# Copyright (c) 2026 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

# A finite sentinel keeps inf out of the lowered graph; exp(-255)
# underflows to exactly 0, so softmax matches an -inf mask bit for bit.
PADDING_MASK_VALUE = -255.0


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
    mask = torch.full((ar_len, ar_len), PADDING_MASK_VALUE)
    mask_cond = torch.arange(ar_len)
    mask.masked_fill_(mask_cond.view(1, ar_len) <= mask_cond.view(ar_len, 1), 0)

    if max_seq_len != ar_len:
        mask = torch.cat(
            [
                torch.ones(ar_len, max_seq_len - ar_len) * PADDING_MASK_VALUE,
                mask,
            ],
            dim=-1,
        )
    mask = mask[None, None, :, :].expand(max_batch_size, 1, ar_len, max_seq_len)
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
    mask = torch.full((ar_len, ar_len), PADDING_MASK_VALUE)
    mask_cond = torch.arange(ar_len)
    mask.masked_fill_(
        (mask_cond.view(1, ar_len) <= mask_cond.view(ar_len, 1))
        & (mask_cond.view(ar_len, 1) - mask_cond.view(1, ar_len) < sliding_window),
        0,
    )

    if max_seq_len != ar_len:
        mask = torch.cat(
            [
                torch.ones(ar_len, max_seq_len - ar_len) * PADDING_MASK_VALUE,
                mask,
            ],
            dim=-1,
        )
    mask = mask[None, None, :, :].expand(max_batch_size, 1, ar_len, max_seq_len)
    return mask
