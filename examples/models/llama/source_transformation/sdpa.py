# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# Example script for exporting Llama2 to flatbuffer

import math
from typing import Tuple

import torch

from executorch.examples.models.llama.attention import KVCache, SDPA


class SDPACustom(torch.nn.Module):
    def __init__(
        self,
        dim: int,
    ):
        super().__init__()
        self.dim = dim

    def forward(
        self,
        input_pos: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bsz,
        seqlen,
        mask,
    ):
        q = q.transpose(1, 2)  # (bs, seqlen, n_local_heads, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Custom op only supports float32 currently. Converting to/from float32 is
        # faster than not having the op.
        input_dtype = q.dtype
        q = q.to(dtype=torch.float)
        k = k.to(dtype=torch.float)
        v = v.to(dtype=torch.float)

        output = torch.ops.llama.custom_sdpa(
            q,
            k,
            v,
            input_pos[0].item(),
            None,  # Attention mask
            0,  # dropout probability. Ignored by the code
            True,  # is_causal
        )
        return output.view(bsz, seqlen, self.dim).to(dtype=input_dtype)


def _replace_sdpa_with_custom_op(module: torch.nn.Module):
    for name, child in module.named_children():
        if isinstance(child, SDPA):
            setattr(
                module,
                name,
                SDPACustom(child.dim),
            )
        else:
            _replace_sdpa_with_custom_op(child)


def replace_sdpa_with_custom_op(module: torch.nn.Module) -> torch.nn.Module:
    from executorch.extension.llm.custom_ops import custom_ops  # noqa

    _replace_sdpa_with_custom_op(module)
    return module


class SDPASimple(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_rep: int,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.n_rep = n_rep

    def forward(
        self,
        input_pos: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bsz,
        seqlen,
        mask,
    ):
        attn_mask = mask[None, None, input_pos]

        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        scale_factor = 1 / math.sqrt(q.size(-1))
        attn_weight = q @ k.transpose(-2, -1) * scale_factor
        attn_weight += attn_mask
        attn_weight = torch.softmax(attn_weight, dim=-1)
        y = attn_weight @ v

        return y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    # TODO: Encounter the bug about source partition, need to investigate more on it.
    # if n_rep == 1:
    #     return hidden_states

    new_kv = []
    batch, n_heads, seqlen, head_dim = hidden_states.shape
    n_heads *= n_rep
    for h in hidden_states[0]:
        new_kv += [h] * n_rep
    return torch.cat(new_kv, 0).reshape(batch, n_heads, seqlen, head_dim)


class SDPAFlex(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        n_rep: int,
    ):
        super().__init__()
        self.dim = dim
        self.n_rep = n_rep

    def forward(
        self,
        input_pos: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bsz,
        seqlen,
        mask,
    ):
        """
        q: (bs, n_heads, seqlen, head_dim)
        k, v: (bs, n_local_heads, seqlen, head_dim)
        """
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        attn_mask = mask[input_pos]

        scale_factor = 1 / math.sqrt(q.size(-1))
        attn_weight = q @ k.transpose(-2, -1) * scale_factor
        attn_weight += attn_mask
        attn_weight = torch.softmax(attn_weight, dim=-1)
        y = attn_weight @ v

        return y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)


def replace_sdpa_with_simple_sdpa(module: torch.nn.Module):
    for name, child in module.named_children():
        if isinstance(child, SDPA):
            setattr(
                module,
                name,
                SDPASimple(child.dim, child.head_dim, child.n_rep),
            )
        else:
            replace_sdpa_with_simple_sdpa(child)
    return module


def replace_sdpa_with_flex_sdpa(module: torch.nn.Module):
    for name, child in module.named_children():
        if isinstance(child, SDPA):
            setattr(
                module,
                name,
                SDPAFlex(child.dim, child.n_rep),
            )
        else:
            replace_sdpa_with_flex_sdpa(child)
    return module


@torch.library.custom_op("coreml::sdpa", mutates_args=())
def sdpa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor
) -> torch.Tensor:
    """Same as F.scaled_dot_product_attention, but with custom op to avoid lowering during dialect conversion."""
    return torch.ops.aten.scaled_dot_product_attention.default(
        q, k, v, attn_mask=attn_mask
    )


@torch.library.register_fake("coreml::sdpa")
def _(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor
) -> torch.Tensor:
    """Fake implementation with the right output shape, which is required for torch.compile/export/fx tracing."""
    expected_shape = list(q.shape)
    expected_shape[-1] = v.shape[-1]
    return q.new_empty(expected_shape)


class SDPACoreML(torch.nn.Module):
    """Similar to SDPASimple, but with coreml custom op to do SDPA calculation."""

    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_rep: int,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.n_rep = n_rep

    def forward(
        self,
        input_pos: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bsz,
        seqlen,
        mask,
    ):
        attn_mask = mask[None, None, input_pos]

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        y = torch.ops.coreml.sdpa(q, k, v, attn_mask)

        return y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)


def replace_sdpa_with_coreml_sdpa(module: torch.nn.Module):
    for name, child in module.named_children():
        if isinstance(child, SDPA):
            setattr(
                module,
                name,
                SDPACoreML(child.dim, child.head_dim, child.n_rep),
            )
        else:
            replace_sdpa_with_coreml_sdpa(child)
    return module


class KVCacheCoreML(torch.nn.Module):
    """
    Rather than k_out[:, :, input_pos] = k_val, use torch.ops.aten.index_put_,
    which can directly translate to CoreML iOS18.silce_update
    """

    def __init__(
        self,
        max_batch_size: int,
        max_context_length: int,
        n_heads: int,
        head_dim: int,
        dtype=torch.float32,
    ):
        super().__init__()
        self.max_context_length = max_context_length
        cache_shape = (max_batch_size, n_heads, max_context_length, head_dim)

        self.max_batch_size = max_batch_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=dtype, device="cpu")
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=dtype, device="cpu")
        )

    def update(
        self, input_pos: torch.Tensor, k_val: torch.Tensor, v_val: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k_out = torch.ops.aten.index_put_(self.k_cache, [None, None, input_pos], k_val)
        v_out = torch.ops.aten.index_put_(self.v_cache, [None, None, input_pos], v_val)
        return k_out, v_out


def replace_kv_cache_with_coreml_kv_cache(module: torch.nn.Module):
    for name, child in module.named_children():
        if isinstance(child, KVCache):
            setattr(
                module,
                name,
                KVCacheCoreML(
                    child.max_batch_size,
                    child.max_context_length,
                    child.n_heads,
                    child.head_dim,
                    child.k_cache.dtype,
                ),
            )
        else:
            replace_kv_cache_with_coreml_kv_cache(child)
    return module


class KVCacheSimple(torch.nn.Module):
    def __init__(
        self,
        max_batch_size: int,
        max_context_length: int,
        n_heads: int,
        head_dim: int,
        dtype=torch.float32,
    ):
        super().__init__()
        cache_shape = (max_batch_size, max_context_length, n_heads, head_dim)
        self.register_buffer(
            "past_k_caches",
            torch.zeros(cache_shape, dtype=dtype, device="cpu"),
            persistent=False,
        )
        self.register_buffer(
            "past_v_caches",
            torch.zeros(cache_shape, dtype=dtype, device="cpu"),
            persistent=False,
        )

    def update(
        self, input_pos: torch.Tensor, k_val: torch.Tensor, v_val: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # can we combine this with KVCacheCoreML?
        k_val = k_val.transpose(1, 2)
        v_val = v_val.transpose(1, 2)
        k_out = torch.ops.aten.index_put_(self.past_k_caches, [None, input_pos], k_val)
        v_out = torch.ops.aten.index_put_(self.past_v_caches, [None, input_pos], v_val)

        k_out = k_out.transpose(1, 2)
        v_out = v_out.transpose(1, 2)
        return k_out, v_out


def replace_kv_cache_with_simple_kv_cache(module: torch.nn.Module):
    for name, child in module.named_children():
        if isinstance(child, KVCache):
            setattr(
                module,
                name,
                KVCacheSimple(
                    child.max_batch_size,
                    child.max_context_length,
                    child.n_heads,
                    child.head_dim,
                    child.k_cache.dtype,
                ),
            )
        else:
            replace_kv_cache_with_simple_kv_cache(child)
    return module


def replace_causal_mask(module: torch.nn.Module):
    for buffer_fqn_name, buffer in module.named_buffers():
        buffer_name = buffer_fqn_name.split(".")[-1]
        if buffer_name == "mask":
            max_context_len = buffer.shape[-1]
            mask = torch.full(
                (max_context_len, max_context_len),
                float("-inf"),
                device="cpu",
            )

            mask = torch.triu(mask, diagonal=1)
            module.register_buffer(buffer_name, mask)
    for _, child in module.named_children():
        replace_causal_mask(child)
    return module
