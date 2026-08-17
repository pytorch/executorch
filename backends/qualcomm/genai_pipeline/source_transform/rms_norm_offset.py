# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""State-dict transforms for RMSNorm weights."""

from __future__ import annotations

from typing import Any, Dict

import torch


def gemma_rmsnorm_offset(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Fold Gemma's implicit ``+1`` into every RMSNorm weight.

    Gemma computes ``(x * w).to(fp16)`` where Llama computes ``x.to(fp16) * w``;
    the static decoder implements the Llama form. See
    https://github.com/huggingface/transformers/pull/29402.
    """
    for k, v in state_dict.items():
        if "norm" not in k:
            continue
        state_dict[k] = v.float() + torch.ones(v.shape, dtype=torch.float32)
    return state_dict
