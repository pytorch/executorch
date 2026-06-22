# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass, fields
from typing import Callable, Dict, Iterator, Optional, Tuple

import torch

from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.rope import (
    hf_precompute_freqs_cis,
    precompute_freqs_cis,
)


ROPE_REGISTRY: Dict[str, Callable] = {}


def register_rope(name: str):
    """Register a rotary embedding function."""

    def decorator(fn: Callable):
        ROPE_REGISTRY[name] = fn
        return fn

    return decorator


ROPE_PRECOMPUTE_REGISTRY: Dict[str, Callable] = {}


def register_rope_precompute(name: str):
    """Register a RopeFreqs precompute method under a precompute-type name.

    The decorated method takes only self and fills the cos/sin fields from
    self.config, so callers just do RopeFreqs(config).compute()."""

    def decorator(method: Callable) -> Callable:
        ROPE_PRECOMPUTE_REGISTRY[name] = method
        return method

    return decorator


def hf_zero_pad_precompute_freqs_cis(
    head_dim, max_context_len, theta, partial_rotary_factor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """HF-style RoPE with zero-padded non-rotated dims, so ROPE_REGISTRY["partial"]
    handles the half-and-half split -- used by gemma4 on both its partial-rotation
    global layers and its full-rotation local layers."""
    rotary_dim = int(head_dim * partial_rotary_factor)
    half_rotary_dim = rotary_dim // 2
    inv_freq_rotary = 1.0 / (
        theta ** (torch.arange(0, rotary_dim, 2).float() / head_dim)
    )
    num_no_rope = head_dim // 2 - half_rotary_dim
    inv_freq = (
        torch.cat([inv_freq_rotary, torch.zeros(num_no_rope)])
        if num_no_rope > 0
        else inv_freq_rotary
    )
    positions = torch.arange(max_context_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


@dataclass
class RopeFreqs:
    """Precomputed rope cos/sin tables plus the config that produced them.

    compute() selects the precompute method for this model and fills the cos/sin
    fields. named_buffers() then yields the (field_name, tensor) pairs the model
    registers directly -- the tensor field names ARE the buffer names.
    Sliding-window models fill the local_* fields; others leave them None.
    """

    config: ModelArgs
    freqs_cos: Optional[torch.Tensor] = None
    freqs_sin: Optional[torch.Tensor] = None
    local_freqs_cos: Optional[torch.Tensor] = None
    local_freqs_sin: Optional[torch.Tensor] = None

    def compute(self) -> "RopeFreqs":
        # gemma4 (global_head_dim set) needs zero-padded partial-rope tables so
        # the buffers stay compatible with the partial apply kernel even on
        # full-rotation local layers -- a case the partial_rotary_factor flag
        # alone can no longer distinguish.
        if self.config.global_head_dim:
            precompute_type = "hf_zero_pad"
        else:
            precompute_type = "hf" if self.config.use_hf_rope else "scaled"
        ROPE_PRECOMPUTE_REGISTRY[precompute_type](self)
        return self

    def named_buffers(self) -> Iterator[Tuple[str, torch.Tensor]]:
        for f in fields(self):
            if f.name == "config":
                continue
            tensor = getattr(self, f.name)
            if tensor is not None:
                yield f.name, tensor

    @register_rope_precompute("scaled")
    def _precompute_freqs_cis(self) -> None:
        config = self.config
        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(
            config.head_dim,
            config.max_context_len,
            config.rope_freq_base,
            config.use_scaled_rope,
            config.rope_scale_factor,
        )

    @register_rope_precompute("hf")
    def _hf_precompute_freqs_cis(self) -> None:
        config = self.config
        freqs_cos, freqs_sin = hf_precompute_freqs_cis(
            config.head_dim,
            config.max_context_len,
            config.rope_freq_base,
            config.partial_rotary_factor,
        )
        self.freqs_cos = freqs_cos[:, : freqs_cos.shape[-1] // 2]
        self.freqs_sin = freqs_sin[:, : freqs_sin.shape[-1] // 2]

        if config.local_rope_theta is not None:
            freqs_cos, freqs_sin = hf_precompute_freqs_cis(
                config.head_dim,
                config.max_context_len,
                config.local_rope_theta,
                config.partial_rotary_factor,
            )
            self.local_freqs_cos = freqs_cos[:, : freqs_cos.shape[-1] // 2]
            self.local_freqs_sin = freqs_sin[:, : freqs_sin.shape[-1] // 2]

    @register_rope_precompute("hf_zero_pad")
    def _hf_zero_pad_precompute_freqs_cis(self) -> None:
        """gemma4: global layers use global_head_dim + partial rope; local
        (sliding) layers use head_dim + full rope, both zero-padded to match
        the partial kernel."""
        config = self.config
        self.freqs_cos, self.freqs_sin = hf_zero_pad_precompute_freqs_cis(
            config.global_head_dim,
            config.max_context_len,
            config.rope_theta,
            config.partial_rotary_factor,
        )
        self.local_freqs_cos, self.local_freqs_sin = hf_zero_pad_precompute_freqs_cis(
            config.head_dim,
            config.max_context_len,
            config.local_rope_theta,
            1.0,
        )


@register_rope("partial")
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


@register_rope("default")
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
