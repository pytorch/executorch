# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Callable, Dict

import torch


ROTARY_EMB_REGISTRY: Dict[str, Callable] = {}


def register_rotary_emb(name: str):
    """Register a rotary embedding function."""

    def decorator(fn: Callable):
        ROTARY_EMB_REGISTRY[name] = fn
        return fn

    return decorator


@register_rotary_emb("partial")
def apply_partial_rotary_emb_single(x, freqs_cos, freqs_sin):
    if x.dim() == 4:
        freqs_cos = freqs_cos[None, None, :, :]
        freqs_sin = freqs_sin[None, None, :, :]
    rotary_dim = freqs_cos.shape[-1] * 2
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x_r, x_i = x_rot[..., : x_rot.shape[-1] // 2], x_rot[..., x_rot.shape[-1] // 2 :]
    x_out_r = x_r * freqs_cos - x_i * freqs_sin
    x_out_i = x_r * freqs_sin + x_i * freqs_cos
    x_rotated = torch.cat([x_out_r, x_out_i], dim=-1)
    return torch.cat([x_rotated, x_pass], dim=-1)


@register_rotary_emb("default")
def apply_rotary_emb_single(x, freqs_cos, freqs_sin):
    # The implementation of RoPE in HuggingFace processes query and key with two half instead of interleaved way.
    # The main difference is stride in StrideSlice op. For interleaved way, stride is two which is not friendly for HTP backend.
    # Ref: https://github.com/huggingface/transformers/issues/25199
    x_r, x_i = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    # broadcast for batch_prefill mode input x
    if x.dim() == 4:
        freqs_cos = freqs_cos[None, None, :, :]
        freqs_sin = freqs_sin[None, None, :, :]
    x_out_r = x_r * freqs_cos - x_i * freqs_sin
    x_out_i = x_r * freqs_sin + x_i * freqs_cos

    x_out = torch.cat([x_out_r, x_out_i], dim=-1)
    return x_out
