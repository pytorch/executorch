# @lint-ignore-every LICENSELINT
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Llama 2 is licensed under the LLAMA 2 Community License,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

# Different RoPE implementations

import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
from executorch.examples.models.llama.model_args import ModelArgs

# ======================== Stock Implementation ========================


def apply_scaling(freqs: torch.Tensor, scale_factor: int, high_freq_factor: int):
    # Values obtained from grid search
    low_freq_factor = 1
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    use_scaled: bool = False,
    scale_factor: Optional[int] = None,
    high_freq_factor: int = 4,
    device: Union[str, torch.device] = "cpu",
):
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim)
    )
    t = torch.arange(end, device=freqs.device)  # pyre-ignore
    if use_scaled:
        assert scale_factor is not None
        freqs = apply_scaling(freqs, scale_factor, high_freq_factor)  # pyre-ignore
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    freqs_cis_ndim = freqs_cis.ndim
    if freqs_cis_ndim == 3:
        # freqs_cis: (seq_len, n_heads, head_dim // 2)
        assert freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1])
        shape = [
            d if (i == ndim - 3 or i == ndim - 2 or i == ndim - 1) else 1
            for i, d in enumerate(x.shape)
        ]
    else:
        # freqs_cis: (seq_len, head_dim // 2)
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def apply_rotary_emb_to_k(
    xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> torch.Tensor:
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    freqs_cos = reshape_for_broadcast(freqs_cos, xk_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xk_r)

    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xk_out.type_as(xk)


# Wrap apply_rotary_emb in a module to enable it to be module swapped out.
class RotaryEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        return xq_out, xk_out


# ======================= HuggingFace Implementation ========================


# Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L77
# and https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py#L242.
# Current only support non-long rope.
def hf_precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float,
    partial_rotary_factor: float = 1.0,
    device: Union[str, torch.device] = "cpu",
):
    # Partial rotary embeddings.
    dim = int(dim * partial_rotary_factor)

    # Short factor scaling.
    freqs = 1.0 / (
        theta
        ** (torch.arange(0, dim, 2, device=device, dtype=torch.int64).float() / dim)
    )
    # TODO: support long factor scaling.

    # pyre-ignore Undefined attribute [16]: `float` has no attribute `device`.
    t = torch.arange(end, device=freqs.device, dtype=torch.int64).type_as(
        freqs  # pyre-ignore
    )
    freqs = torch.outer(t, freqs).float()  # pyre-ignore
    emb = torch.cat((freqs, freqs), dim=-1)
    freqs_cos = torch.cos(emb)
    freqs_sin = torch.sin(emb)
    return freqs_cos, freqs_sin


# Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L135
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def hf_apply_rotary_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = torch.cat([(q_rot * cos) + (rotate_half(q_rot) * sin), q_pass], dim=-1)
    k_embed = torch.cat([(k_rot * cos) + (rotate_half(k_rot) * sin), k_pass], dim=-1)
    return q_embed, k_embed


def hf_apply_rotary_emb_to_k(k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the key tensors.

    Args:
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of k. Similarly, if k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `torch.Tensor` the key tensor rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return k_embed


class Rope(torch.nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params

        # Choose the appropriate RoPE implementation
        if self.params.use_hf_rope:
            self.precompute_freqs_cis = partial(
                hf_precompute_freqs_cis,
                partial_rotary_factor=self.params.partial_rotary_factor,
                device=getattr(self.params, "device", "cpu"),
            )
            self.apply_rotary_emb = hf_apply_rotary_emb
        else:
            self.precompute_freqs_cis = partial(
                precompute_freqs_cis,
                use_scaled=self.params.use_scaled_rope,
                scale_factor=self.params.rope_scale_factor,
                high_freq_factor=self.params.high_freq_factor,
                device=getattr(self.params, "device", "cpu"),
            )
            self.apply_rotary_emb = RotaryEmbedding()

        # Precompute frequencies
        freqs_cos, freqs_sin = self.precompute_freqs_cis(
            self.params.head_dim,
            (
                self.params.max_context_len  # Normal llama2.
                if self.params.ffn_dim_multiplier is None
                else self.params.max_context_len * 2  # Sharded checkpoint.
            ),
            self.params.rope_freq_base,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        return self.apply_rotary_emb(q, k, freqs_cos, freqs_sin)

    def get_freqs(self, input_pos: Optional[torch.Tensor], seq_len: int):
        """
        Get the precomputed frequencies for the given input position and sequence length.

        Args:
            input_pos (torch.Tensor): The input position tensor.
            seq_len (int): The sequence length.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The precomputed frequencies for the given input position and sequence length.
        """
        if self.params.use_kv_cache:
            assert (
                input_pos is not None
            ), "input_pos must be provided when use_kv_cache is True"

            if self.params.enable_dynamic_shape:
                # when KV cache is used, seqlen is most likely 1. We want to slice from the start_pos.
                input_pos_item = input_pos[-1].item()
                torch._check_is_size(input_pos_item)
                torch._check(input_pos_item < self.params.max_context_len)
                # pyre-ignore: Incompatible parameter type [6]: torch.narrow does expect int or Tensor
                freqs_cos = self.freqs_cos.narrow(0, input_pos_item, seq_len)
                # pyre-ignore: Incompatible parameter type [6]
                freqs_sin = self.freqs_sin.narrow(0, input_pos_item, seq_len)
            else:
                # When not using dynamic shape, use of the .item results in
                # symints, due to querying the data from tensor.
                # this path avoids that for mps backend, although probably mps backend
                # can support dynamic shape?
                freqs_cos = self.freqs_cos[input_pos]
                freqs_sin = self.freqs_sin[input_pos]

        else:
            assert input_pos is None, "input_pos is unused when use_kv_cache is False"
            freqs_cos = self.freqs_cos[:seq_len]
            freqs_sin = self.freqs_sin[:seq_len]
        return freqs_cos, freqs_sin

    def get_freqs_using_indices(self, indices: torch.Tensor):
        """
        Get the precomputed frequencies for given input indices.

        Args:
            indices (torch.Tensor): The input indices tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The precomputed frequencies for given input indices.
        """
        return self.freqs_cos[indices], self.freqs_sin[indices]
