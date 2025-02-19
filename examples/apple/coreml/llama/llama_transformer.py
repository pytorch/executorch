# @lint-ignore-every LICENSELINT
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

# Please refer to README.md in the same folder for more information.

from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from executorch.examples.models.llama.llama_transformer import RMSNorm

from executorch.examples.models.llama.rope import (
    hf_apply_rotary_emb,
    hf_precompute_freqs_cis,
    precompute_freqs_cis,
    RotaryEmbedding,
)

from torch import nn


# These are just to prevent to_edge from decomposing SDPA
# A better method is to use the to_edge_transform_and_lower API for CoreML
# and not decompose SDPA
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


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class ModelArgs:
    dim: int = 2048
    n_layers: int = 16
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 128256
    hidden_dim: Optional[int] = None
    head_dim: Optional[int] = None  # Optional customized head_dim
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 1
    max_seq_len: int = 128
    max_context_len: int = 2048
    moe: bool = False  # True to enable the MoE (Mixture of Experts)
    num_experts: int = 8  # Number of experts
    num_activated_experts: int = 2  # Number of experts to activate

    # Generate logits for all inputs. When it's True, it would take big memory usage
    # at runtime. Enable it only necessary (e.g., use perplexity tools that requires
    # logits for all input tokens.)
    generate_full_logits: bool = False
    # A dictionary mapping from pruned token-id to original token-id
    input_prune_map: Optional[Dict[int, int]] = None
    # A dictionary mapping from pruned token-id to original token-id
    output_prune_map: Optional[Dict[int, int]] = None
    use_hf_rope: bool = False  # Use HuggingFace's RoPE implementation
    rope_theta: Optional[float] = (
        None  # The official name to override self.rope_freq_base.
    )
    rope_freq_base: float = 10000.0  # The base frequency for RoPE. Keep it for BC.
    use_scaled_rope: bool = True  # Use scaled RoPE, introduced in llama3.1.
    # Additional Model Metadata needed at runtime
    rope_scale_factor: int = 8
    bos_idx: int = 1
    eos_idx: int = 3
    bos_count: int = -1  # i.e., a single EOS is used as BOS
    eos_count: int = 2

    quantization_args: Optional[dict] = None
    lora_args: Optional[dict] = None

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        # rope_theta overrides rope_freq_base since it's the official name.
        if self.rope_theta is not None:
            self.rope_freq_base = self.rope_theta

        if self.hidden_dim is None:
            # If hidden_dim is not explicitly set in the ModelArgs,
            # then calculate implicitly based on dim and also multiple of `args.multiple_of`
            multiple_of = self.multiple_of
            hidden_dim = 4 * self.dim
            hidden_dim = int(2 * hidden_dim / 3)
            if self.ffn_dim_multiplier is not None:
                hidden_dim = int(self.ffn_dim_multiplier * hidden_dim)
            self.hidden_dim = find_multiple(hidden_dim, multiple_of)

        if self.head_dim is None:
            self.head_dim = self.dim // self.n_heads


class Rope(torch.nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        if self.params.use_hf_rope:
            self.precompute_freqs_cis = hf_precompute_freqs_cis
        else:
            self.precompute_freqs_cis = partial(
                precompute_freqs_cis, use_scaled=self.params.use_scaled_rope
            )
        freqs_cos, freqs_sin = self.precompute_freqs_cis(
            self.params.head_dim,
            (
                self.params.max_context_len  # Normal llama2.
                if self.params.ffn_dim_multiplier is None
                else self.params.max_context_len * 2  # Sharded checkpoint.
            ),
            self.params.rope_freq_base,
            scale_factor=8,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        if self.params.use_hf_rope:
            self.apply_rotary_emb = hf_apply_rotary_emb
        else:
            self.apply_rotary_emb = RotaryEmbedding()

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
        assert (
            input_pos is not None
        ), "input_pos must be provided when use_kv_cache is True"
        input_pos_item = input_pos[-1].item()

        # CoreML partitioner is not picking up _check_is_size
        # So instead use _check as workaround.  Should be easy fix for partitioner
        # torch._check_is_size(input_pos_item)
        torch._check(input_pos_item >= 0)
        torch._check(input_pos_item + seq_len <= self.params.max_seq_len)
        # pyre-ignore: Incompatible parameter type [6]: torch.narrow does expect int or Tensor
        freqs_cos = self.freqs_cos.narrow(0, input_pos_item, seq_len)
        # pyre-ignore: Incompatible parameter type [6]
        freqs_sin = self.freqs_sin.narrow(0, input_pos_item, seq_len)

        return freqs_cos, freqs_sin


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.hidden_dim is not None
        hidden_dim: int = args.hidden_dim
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ConditionalFeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        hidden_dim = args.hidden_dim
        if hidden_dim is None:
            # If hidden_dim is not explicitly set in the ModelArgs,
            # then calculate implicitly based on dim and also multiple of `args.multiple_of`
            multiple_of = args.multiple_of
            hidden_dim = 4 * self.dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Parameter(torch.randn(args.num_experts, hidden_dim, self.dim))
        self.w2 = nn.Parameter(torch.randn(args.num_experts, hidden_dim, self.dim))
        self.w3 = nn.Parameter(torch.randn(args.num_experts, hidden_dim, self.dim))
        self.num_experts = args.num_experts

    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        w1_weights = self.w1[expert_indices].transpose(-1, -2)  # [T, A, D, D]
        w3_weights = self.w3[expert_indices].transpose(-1, -2)  # [T, A, D, D]
        w2_weights = self.w2[expert_indices]  # [T, A, D, D]
        x1 = F.silu(torch.einsum("ti,taio -> tao", x, w1_weights))
        x3 = torch.einsum("ti, taio -> tao", x, w3_weights)
        expert_outs = torch.einsum("tao, taoi -> tai", (x1 * x3), w2_weights)
        return expert_outs


class MOEFeedForward(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.gate = nn.Linear(config.dim, config.num_experts, bias=False)
        self.cond_ffn = ConditionalFeedForward(config)
        self.dim = config.dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.dim)
        # T = num_tokens, E = num_experts, D = hidden dim, A = activated experts
        # x: [T, D]
        scores = self.gate(x)  # [T, E]
        expert_weights, expert_indices = torch.topk(scores, 2, dim=-1)  # [T, A], [T, A]
        expert_weights = expert_weights.softmax(dim=-1)  # [T, A]
        expert_outs = self.cond_ffn(x, expert_indices)
        return torch.einsum("tai,ta -> ti", expert_outs, expert_weights)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int, rope: Rope):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = self.n_heads if args.n_kv_heads is None else args.n_kv_heads

        assert self.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = self.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.head_dim
        self.max_batch_size = args.max_batch_size
        self.max_seq_len = args.max_seq_len
        self.dim = args.dim
        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        self.layer_id = layer_id

        self.rope = rope

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_mask: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape
        # QKV
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        # We need view_copy elimination
        q = q.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        q, k = self.rope.forward(q, k, freqs_cos, freqs_sin)

        q = q.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        new_k = k
        new_v = v

        k = torch.concat([k_cache, k], dim=2)
        v = torch.concat([v_cache, v], dim=2)

        # grouped multiquery attention: expand out keys and values
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        output = torch.ops.coreml.sdpa(q, k, v, attn_mask)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        output = self.wo(output)

        return output, new_k, new_v


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, rope: Rope):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.head_dim
        self.attention = Attention(args, layer_id, rope)
        if args.moe:
            self.block_sparse_moe = MOEFeedForward(args)
        else:
            self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x,
        freqs_cos,
        freqs_sin,
        k_cache,
        v_cache,
        attn_mask,
    ):  # x: 1xN
        norm_emb = self.attention_norm(x)
        h, new_k, new_v = self.attention.forward(
            norm_emb, freqs_cos, freqs_sin, k_cache, v_cache, attn_mask
        )

        h = x + h
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, new_k, new_v


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.rope = Rope(params)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params, self.rope))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.generate_full_logits = params.generate_full_logits
        self.max_seq_len = params.max_seq_len
        self.input_prune_map = params.input_prune_map
        self.output_prune_map = params.output_prune_map

    def forward(
        self,
        tokens: torch.LongTensor,  # tokens
        input_pos: torch.LongTensor,
        input_length: torch.LongTensor,  # input_length
        k_cache: torch.FloatTensor,
        v_cache: torch.FloatTensor,
        attn_mask: torch.LongTensor,
        h: Optional[torch.FloatTensor] = None,  # embeddings
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (tokens is None) ^ (h is not None):
            raise ValueError(
                "You cannot specify both tokens and h at the same time, and must specify either one"
            )
        if tokens is not None and h is None:
            h = self.tok_embeddings(tokens)
        seqlen = h.shape[1]
        freqs_cos, freqs_sin = self.rope.get_freqs(input_pos, seqlen)

        k_out = []
        v_out = []
        for i, layer in enumerate(self.layers):
            h, new_k, new_v = layer(
                h,
                freqs_cos,
                freqs_sin,
                k_cache[i, :, :, :, :],
                v_cache[i, :, :, :, :],
                attn_mask,
            )
            k_out.append(new_k)
            v_out.append(new_v)

        if not self.generate_full_logits:
            # Only the last logit is used for the new generated token
            h = h[:, input_length - 1, :]

        h = self.norm(h)

        logits = self.output(h)

        return (
            logits,
            torch.stack(k_out, dim=0),
            torch.stack(v_out, dim=0),
        )


class InputManager:
    def __init__(
        self,
        model_args: ModelArgs,
        seq_length,
        dtype=torch.float16,
        minus_infinity=-torch.inf,
    ):
        self.n_layers = model_args.n_layers
        self.max_batch_size = model_args.max_batch_size
        self.n_kv_heads = model_args.n_kv_heads
        self.head_dim = model_args.head_dim

        self.seq_length = seq_length
        self.max_seq_length = model_args.max_seq_len

        self.k_cache = torch.zeros(
            self.get_cache_shape(self.max_seq_length - self.seq_length)
        ).to(dtype)
        self.v_cache = torch.zeros(
            self.get_cache_shape(self.max_seq_length - self.seq_length)
        ).to(dtype)

        attn_cache = minus_infinity * torch.ones(
            seq_length, self.max_seq_length - self.seq_length
        )  # attn for past tokens
        attn_seq = torch.triu(
            minus_infinity * torch.ones(self.seq_length, self.seq_length), diagonal=1
        )  # attn for current tokens
        self.attn_mask = torch.concat([attn_cache, attn_seq], dim=-1).to(dtype)
        assert self.attn_mask.shape == (self.seq_length, self.max_seq_length)

        self.input_pos = 0

    def get_cache_shape(self, length):
        return (
            self.n_layers,
            self.max_batch_size,
            self.n_kv_heads,
            length,
            self.head_dim,
        )

    def update(self, input_length, new_k_cache, new_v_cache):
        assert new_k_cache.shape == self.get_cache_shape(self.seq_length)
        assert new_v_cache.shape == self.get_cache_shape(self.seq_length)

        self.k_cache[:, :, :, (self.input_pos) : (self.input_pos + input_length), :] = (
            new_k_cache[:, :, :, 0:input_length, :]
        )
        self.v_cache[:, :, :, (self.input_pos) : (self.input_pos + input_length), :] = (
            new_v_cache[:, :, :, 0:input_length, :]
        )
        self.attn_mask[:, (self.input_pos) : (self.input_pos + input_length)] = 0.0
        self.input_pos += input_length

    def get_inputs(self, tokens: List[int]):
        input_length = len(tokens)
        assert input_length <= self.seq_length

        return (
            # tokens
            torch.concat(
                [
                    torch.tensor(tokens, dtype=torch.int64),
                    torch.zeros(self.seq_length - input_length, dtype=torch.int64),
                ],
                axis=-1,
            ).reshape(1, -1),
            # input_pos
            torch.tensor([self.input_pos], dtype=torch.long),
            # input_length
            torch.tensor([input_length], dtype=torch.long),
            # k_cache
            self.k_cache,
            # v_cache
            self.v_cache,
            # attn_mask
            self.attn_mask,
        )

    def get_inputs_and_remaining_tokens(self, tokens: List[int]):
        processed_tokens = min(self.seq_length, len(tokens))
        return (
            self.get_inputs(tokens[0:processed_tokens]),
            processed_tokens,
            tokens[processed_tokens:],
        )
