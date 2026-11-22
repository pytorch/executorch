# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Activation format of the NVFP4 attention projections (CUDA).

``export_solo.py --nvfp4-attn-act {fp4,mxfp8,bf16}`` keeps the attention
projection weights (fused ``qko`` / ``qkv`` or ``q`` / ``k`` / ``og``, ``v``
and ``o``) NVFP4 and sets their ``CudaNVFP4Tensor.act_dtype``:

- ``fp4``: unchanged (W4A4 where FP4 tensor cores exist).
- ``mxfp8``: MXFP8 x MXFP8 prefill with the weights converted in registers;
  decode W4A16 (or the INT8-activation W4A8 GEMV without FP4 conversion).
- ``bf16``: weight-only W4A16.

FFN, embedding and lm_head keep their paths.
"""

import torch.nn as nn

from executorch.backends.cuda.nvfp4_tensor import ACT_DTYPES, CudaNVFP4Tensor

ATTN_PROJECTIONS = (
    "qko_proj",
    "qkv_proj",
    "q_proj",
    "k_proj",
    "og_proj",
    "v_proj",
    "o_proj",
)


def mark_attention_act(model: nn.Module, act_dtype: str) -> int:
    """Set ``act_dtype`` on every NVFP4 attention projection; returns the count."""
    if act_dtype not in ACT_DTYPES:
        raise ValueError(f"act_dtype must be one of {ACT_DTYPES}")
    count = 0
    for layer in model.layers:
        attn = layer.self_attn
        for name in ATTN_PROJECTIONS:
            linear = getattr(attn, name, None)
            weight = getattr(linear, "weight", None)
            if isinstance(weight, CudaNVFP4Tensor):
                linear.weight = nn.Parameter(
                    weight.with_act_dtype(act_dtype), requires_grad=False
                )
                count += 1
    print(
        f"[muse_glimmer cuda] NVFP4 attention projections act_dtype={act_dtype}: "
        f"{count} linears"
    )
    return count
