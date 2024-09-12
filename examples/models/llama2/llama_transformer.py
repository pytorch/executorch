# @lint-ignore-every LICENSELINT
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Llama 2 is licensed under the LLAMA 2 Community License,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

# Please refer to README.md in the same folder for more information.

from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from executorch.examples.models.llama2.rope import (
    apply_rotary_emb,
    hf_apply_rotary_emb,
    hf_precompute_freqs_cis,
    precompute_freqs_cis,
)

from torch import nn


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048
    moe: bool = False  # True to enable the MoE (Mixture of Experts)
    num_experts: int = 8  # Number of experts
    num_activated_experts: int = 2  # Number of experts to activate
    use_kv_cache: bool = False  # Use key/value cache
    use_sdpa_with_kv_cache_op: bool = (
        False  # Use custom sdpa op that updates kv cache in-place
    )
    # Generate logits for all inputs. When it's True, it would take big memory usage
    # at runtime. Enable it only necessary (e.g., use perplexity tools that requires
    # logits for all input tokens.)
    generate_full_logits: bool = False
    enable_dynamic_shape: bool = False  # export model with dynamic shape support
    use_hf_rope: bool = False  # Use HuggingFace's RoPE implementation
    rope_theta: Optional[float] = (
        None  # The official name to override self.rope_freq_base.
    )
    rope_freq_base: float = 10000.0  # The base frequency for RoPE. Keep it for BC.
    use_scaled_rope: bool = False  # Use scaled RoPE, introduced in llama3.1.
    # Additional Model Metadata needed at runtime
    bos_idx: int = 1
    eos_idx: int = 3
    bos_count: int = -1  # i.e., a single EOS is used as BOS
    eos_count: int = 2

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        # rope_theta overrides rope_freq_base since it's the official name.
        if self.rope_theta is not None:
            self.rope_freq_base = self.rope_theta

        if self.use_sdpa_with_kv_cache_op:
            assert self.use_kv_cache, "use_sdpa_with_kv_cache_op requires use_kv_cache"

        if self.hidden_dim is None:
            # If hidden_dim is not explicitly set in the ModelArgs,
            # then calculate implicitly based on dim and also multiple of `args.multiple_of`
            multiple_of = self.multiple_of
            hidden_dim = 4 * self.dim
            hidden_dim = int(2 * hidden_dim / 3)
            if self.ffn_dim_multiplier is not None:
                hidden_dim = int(self.ffn_dim_multiplier * hidden_dim)
            self.hidden_dim = find_multiple(hidden_dim, multiple_of)


class KVCache(nn.Module):
    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_dim: int,
        transpose_cache: bool,
        enable_dynamic_shape: bool,
        dtype=torch.float32,
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        if transpose_cache:
            cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        else:
            cache_shape = (max_batch_size, max_seq_length, n_heads, head_dim)

        self.max_batch_size = max_batch_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.transpose_cache = transpose_cache
        self.enable_dynamic_shape = enable_dynamic_shape
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=dtype, device="cpu")
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=dtype, device="cpu")
        )

    def update(
        self, input_pos: torch.Tensor, k_val: torch.Tensor, v_val: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # input_pos: [S], k_val: [B, H, S, D] or [B, S, H, D] depending on transpose_cache
        if self.enable_dynamic_shape:
            start_pos = input_pos[-1].item()
            torch._check_is_size(start_pos)
            torch._check(start_pos < self.max_seq_length)
            seq_length = k_val.size(2)
            # Replace the entry in the cache for this token
            # The following lines are equivalent to:
            # cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            # cache_v[:bsz, start_pos : start_pos + seqlen] = xv
            # We use .narrow() here to make the compiler happy
            # pyre-ignore: Incompatible parameter type [6]
            narrowed_k = self.k_cache.narrow(2, start_pos, seq_length)
            # pyre-ignore: Incompatible parameter type [6]
            narrowed_v = self.v_cache.narrow(2, start_pos, seq_length)

            narrowed_k.copy_(k_val)
            narrowed_v.copy_(v_val)
            return self.k_cache, self.v_cache
        else:
            k_out = self.k_cache
            v_out = self.v_cache
            k_out[:, :, input_pos] = k_val
            v_out[:, :, input_pos] = v_val

            return k_out, v_out


class SDPA(nn.Module):
    def __init__(
        self,
        kv_cache: KVCache,
        dim: int,
        head_dim: int,
        n_rep: int,
        max_seq_len: int,
        enable_dynamic_shape: bool,
    ):
        super().__init__()
        self.kv_cache = kv_cache
        self.dim = dim
        self.head_dim = head_dim
        self.n_rep = n_rep
        self.max_seq_len = max_seq_len
        self.enable_dynamic_shape = enable_dynamic_shape

    def forward(
        self,
        input_pos: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bsz,
        seqlen,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        q = q.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k, v = self.kv_cache.update(input_pos, k, v)
        if self.enable_dynamic_shape:
            start_pos = input_pos[-1].item()
            torch._check_is_size(start_pos)
            torch._check(start_pos < self.max_seq_len)
            seq_length = q.size(2)
            # pyre-ignore: Incompatible parameter type [6]
            attn_mask = mask.narrow(0, start_pos, seq_length)
        else:
            attn_mask = mask[None, None, input_pos]

        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)

        return y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.use_kv_cache = args.use_kv_cache
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.max_batch_size = args.max_batch_size
        self.max_seq_len = args.max_seq_len
        self.dim = args.dim
        # args.dim = 4096, args.n_heads = 32, self.head_dim = 4096 / 32 = 125
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.layer_id = layer_id

        causal_mask = torch.tril(
            torch.ones(
                self.max_seq_len,
                self.max_seq_len,
                dtype=torch.bool,
                device="cpu",
            )
        )
        self.register_buffer("mask", causal_mask, persistent=False)

        if self.use_kv_cache:
            self.kv_cache = KVCache(
                args.max_batch_size,
                args.max_seq_len,
                self.n_kv_heads,
                self.head_dim,
                not args.use_sdpa_with_kv_cache_op,  # if we are using the custom op dont transpose the cache. Expect untransposed q k v
                args.enable_dynamic_shape,
            )
            self.SDPA = SDPA(
                kv_cache=self.kv_cache,
                dim=self.dim,
                head_dim=self.head_dim,
                n_rep=self.n_rep,
                max_seq_len=self.max_seq_len,
                enable_dynamic_shape=args.enable_dynamic_shape,
            )
        if args.use_hf_rope:
            self.apply_rotary_emb = hf_apply_rotary_emb
        else:
            self.apply_rotary_emb = apply_rotary_emb

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen, _ = x.shape

        # QKV
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        # We need view_copy elimination
        q = q.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        q, k = self.apply_rotary_emb(q, k, freqs_cos, freqs_sin)

        if self.use_kv_cache:
            assert input_pos is not None
            output = self.SDPA(input_pos, q, k, v, bsz, seqlen, self.mask)
            return self.wo(output)

        q = q.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # grouped multiquery attention: expand out keys and values
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        assert hasattr(self, "mask")

        mask = self.mask[:seqlen, :seqlen]

        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        output = self.wo(output)

        return output


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


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.use_kv_cache = args.use_kv_cache
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args, layer_id)
        if args.moe:
            self.block_sparse_moe = MOEFeedForward(args)
        else:
            self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin, input_pos=None):  # x: 1xN
        h = self.attention.forward(
            self.attention_norm(x), freqs_cos, freqs_sin, input_pos
        )

        h = x + h
        if hasattr(self, "block_sparse_moe"):
            out = h + self.block_sparse_moe(self.ffn_norm(h))
        else:
            out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.use_kv_cache = params.use_kv_cache
        self.generate_full_logits = params.generate_full_logits
        self.max_seq_len = params.max_seq_len
        if params.use_hf_rope:
            self.precompute_freqs_cis = hf_precompute_freqs_cis
        else:
            self.precompute_freqs_cis = partial(
                precompute_freqs_cis, use_scaled=params.use_scaled_rope
            )
        freqs_cos, freqs_sin = self.precompute_freqs_cis(
            params.dim // params.n_heads,
            (
                params.max_seq_len  # Normal llama2.
                if params.ffn_dim_multiplier is None
                else params.max_seq_len * 2  # Sharded checkpoint.
            ),
            params.rope_freq_base,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        tokens: Optional[torch.LongTensor] = None,  # tokens
        input_pos: Optional[
            torch.LongTensor
        ] = None,  # Scalar tensor indicating size of window of the caches
        h: Optional[torch.FloatTensor] = None,  # embeddings
    ) -> torch.Tensor:
        if (tokens is None) ^ (h is not None):
            raise ValueError(
                "You cannot specify both tokens and h at the same time, and must specify either one"
            )
        if tokens is not None and h is None:
            h = self.tok_embeddings(tokens)
        seqlen = h.shape[1]

        if self.use_kv_cache:
            assert (
                input_pos is not None
            ), "input_pos must be provided when use_kv_cache is True"

            if self.params.enable_dynamic_shape:
                # when KV cache is used, seqlen is most likely 1. We want to slice from the start_pos.
                input_pos_item = input_pos[-1].item()
                torch._check_is_size(input_pos_item)
                torch._check(input_pos_item < self.params.max_seq_len)
                # pyre-ignore: Incompatible parameter type [6]: torch.narrow does expect int or Tensor
                freqs_cos = self.freqs_cos.narrow(0, input_pos_item, seqlen)
                # pyre-ignore: Incompatible parameter type [6]
                freqs_sin = self.freqs_sin.narrow(0, input_pos_item, seqlen)
            else:
                # When not using dynamic shape, use of the .item results in
                # symints, due to querying the data from tensor.
                # this path avoids that for mps backend, although probably mps backend
                # can support dynamic shape?
                freqs_cos = self.freqs_cos[input_pos]
                freqs_sin = self.freqs_sin[input_pos]

        else:
            assert input_pos is None, "input_pos is unused when use_kv_cache is False"
            freqs_cos = self.freqs_cos[:seqlen]
            freqs_sin = self.freqs_sin[:seqlen]

        for layer in self.layers:
            h = layer(
                h,
                freqs_cos,
                freqs_sin,
                input_pos,
            )

        if not self.generate_full_logits:
            # Only the last logit is used for the new generated token
            h = h[:, -1, :]

        h = self.norm(h)

        logits = self.output(h)
        return logits
