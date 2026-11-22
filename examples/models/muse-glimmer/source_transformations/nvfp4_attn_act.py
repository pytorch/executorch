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
- ``mxfp8``: MXFP8 x MXFP8 prefill with the weights converted in registers
  (native FP4 conversion, SM120; BF16 activations elsewhere); decode runs the
  W4A16 GEMV, timed against the INT8-activation W4A8 GEMV without FP4
  conversion.
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
    """Set ``act_dtype`` on every NVFP4 attention projection; returns the count.

    Raises if the model has no NVFP4 attention projection to mark.
    """
    if act_dtype not in ACT_DTYPES:
        raise ValueError(f"act_dtype must be one of {ACT_DTYPES}")
    count = 0
    for layer in model.layers:
        for name, linear in layer.self_attn.named_children():
            if name in ATTN_PROJECTIONS and isinstance(linear.weight, CudaNVFP4Tensor):
                linear.weight = nn.Parameter(
                    linear.weight.with_act_dtype(act_dtype), requires_grad=False
                )
                count += 1
    if count == 0:
        raise ValueError("No NVFP4 attention projection to set act_dtype on")
    print(
        f"[muse_glimmer cuda] NVFP4 attention projections act_dtype={act_dtype}: "
        f"{count} linears"
    )
    return count
