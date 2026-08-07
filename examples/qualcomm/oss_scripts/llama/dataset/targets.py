# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from executorch.examples.qualcomm.oss_scripts.llama.dataset.constants import (
    LABEL_IGNORE_INDEX,
)


def make_causal_labels(tokens: List[int], max_context_len: int) -> List[int]:
    """
    Shifted next-token labels: labels[i] = tokens[i+1]; padding positions will be ignored.

    """
    n = min(len(tokens) - 1, max_context_len - 1)
    labels = [LABEL_IGNORE_INDEX] * max_context_len
    if n > 0:
        labels[:n] = tokens[1 : n + 1]
    return labels


def make_conversation_labels(
    tokens: List[int],
    assistant_mask: List[int],
    max_context_len: int,
) -> List[int]:
    """
    Conversation labels: only assistant-turn positions supervised; others are ignored.

    Same pre-shift convention as make_causal_labels. assistant_mask[i] == 1 marks
    token i as an assistant-response token. Callers must guarantee the mask supervises
    at least one token; an all-zero mask yields all-ignored labels here.
    """
    n = min(len(tokens), max_context_len)
    labels = [LABEL_IGNORE_INDEX] * max_context_len
    for i in range(n - 1):
        if i < len(assistant_mask) and assistant_mask[i]:
            labels[i - 1] = tokens[i]
    return labels
