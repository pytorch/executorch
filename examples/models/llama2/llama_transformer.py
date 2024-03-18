# @lint-ignore-every LICENSELINT
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Llama 2 is licensed under the LLAMA 2 Community License,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

# Please refer to README.md in the same folder for more information.

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

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
    rope_freq_base: float = 10000.0  # The base frequency for RoPE
    # Additional Model Metadata needed at runtime
    bos_idx: int = 1
    eos_idx: int = 3
    bos_count: int = -1  # i.e., a single EOS is used as BOS
    eos_count: int = 2

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        if self.use_sdpa_with_kv_cache_op:
            assert self.use_kv_cache, "use_sdpa_with_kv_cache_op requires use_kv_cache"


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def precompute_freqs_cis(dim: int, end: int, theta: float):
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device="cpu")[: (dim // 2)].float() / dim)
    )
    t = torch.arange(end, device=freqs.device)  # pyre-ignore
    freqs = torch.outer(t, freqs).float()  # pyre-ignore
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
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
        # args.dim = 4096, args.n_heads = 32, self.head_dim = 4096 / 32 = 125
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.use_sdpa_with_kv_cache_op = args.use_sdpa_with_kv_cache_op
        self.layer_id = layer_id

        mask = torch.full(
            (1, 1, args.max_seq_len, args.max_seq_len),
            float("-inf"),
            device="cpu",
        )

        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)

        # This is what we would use if ExecuTorch could support mutable buffers. We can't at this time, so instead
        # what is done is this module takes in the cache as io.
        # self.cache_k = torch.zeros(
        #     (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        # )
        # self.cache_v = torch.zeros(
        #     (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        # )
        self.kv_cache_sizes = [
            args.max_batch_size,
            args.max_seq_len,
            self.n_kv_heads,
            self.head_dim,
        ]

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        start_pos: Optional[int] = None,
        cache_k: Optional[torch.Tensor] = None,
        # if use_sdpa_with_kv_cache_op
        # shape: (num_layers, args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        # otherwise
        # shape: (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        cache_v: Optional[torch.Tensor] = None,
        # if use_sdpa_with_kv_cache_op
        # shape: (num_layers, args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        # otherwise
        # shape: (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
    ):
        bsz, seqlen, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # We need view_copy elimination
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        if self.use_kv_cache:
            assert start_pos is not None
            assert cache_k is not None and cache_v is not None

            # TODO(T180671810)
            # Refactor this code to make custom op based
            # SDPA into a separate optimized attention module
            if self.use_sdpa_with_kv_cache_op:
                from .custom_ops.sdpa_with_kv_cache import sdpa_with_kv_cache  # noqa

                output = torch.ops.llama.sdpa_with_kv_cache(
                    xq,
                    xk,
                    xv,
                    cache_k,
                    cache_v,
                    self.layer_id,
                    start_pos,
                    seqlen,
                )
                output = output.view(bsz, seqlen, -1)
                output = self.wo(output)
                return output, cache_k, cache_v
            else:
                # Replace the entry in the cache for this token
                # The following lines are equivalent to:
                # cache_k[:bsz, start_pos : start_pos + seqlen] = xk
                # cache_v[:bsz, start_pos : start_pos + seqlen] = xv
                # We use .narrow() here to make the compiler happy
                narrowed_k = cache_k[:bsz].narrow(1, start_pos, seqlen)
                narrowed_v = cache_v[:bsz].narrow(1, start_pos, seqlen)

                narrowed_k.copy_(xk)
                narrowed_v.copy_(xv)

                keys = cache_k[:bsz].narrow(1, 0, start_pos + seqlen)
                values = cache_v[:bsz].narrow(1, 0, start_pos + seqlen)
        else:
            keys = xk
            values = xv

        # grouped multiquery attention: expand out keys and values
        keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        assert hasattr(self, "mask")
        mask = self.mask[:, :, :seqlen, :seqlen]

        # this is needed to support xnnpack which requires mask shape to be 2d.
        # this is a temporary workaround. once we update xnnpack we should be able to handle this.
        # shape before: [1, 1, l, s], after: [l, s]
        # we make sure to specify the dimensions to be squeezed [0, 1] to ensure that the output
        # tensor will be 2-dimensional, regarldess of the values of l & s
        mask = torch.squeeze(mask, [0, 1])

        output = F.scaled_dot_product_attention(
            xq, keys, values, attn_mask=mask, dropout_p=0.0
        )

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        output = self.wo(output)

        if self.use_kv_cache:
            return output, cache_k, cache_v
        else:
            return output, None, None


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.dim
        hidden_dim = args.hidden_dim
        if hidden_dim is None:
            # If hidden_dim is not explicitly set in the ModelArgs,
            # then calculate implicitly based on dim and also multiple of `args.multiple_of`
            multiple_of = args.multiple_of
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

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

    def forward(
        self, x, freqs_cos, freqs_sin, start_pos=None, cache_k=None, cache_v=None
    ):  # x: 1xN
        h, cache_k, cache_v = self.attention.forward(
            self.attention_norm(x),
            freqs_cos,
            freqs_sin,
            start_pos,
            cache_k,
            cache_v,
        )

        h = x + h
        if hasattr(self, "block_sparse_moe"):
            out = h + self.block_sparse_moe(self.ffn_norm(h))
        else:
            out = h + self.feed_forward(self.ffn_norm(h))
        return out, cache_k, cache_v


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

        freqs_cos, freqs_sin = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len,
            params.rope_freq_base,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: Optional[
            torch.Tensor
        ] = None,  # Scalar tensor indicating size of window of the caches
        cache_k: Optional[
            torch.Tensor
        ] = None,  # n_layers long, it should be a list of tensors to accommodate the potential size difference among attention layers. The current implementation is overly simplified.
        cache_v: Optional[torch.Tensor] = None,  # n_layers long
    ) -> Union[
        torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]
    ]:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        if self.use_kv_cache:
            assert (
                cache_k is not None and cache_v is not None and start_pos is not None
            ), "Caches and start_pos must be provided when use_kv_cache is True"
            assert (
                cache_k.size(0) == self.n_layers
            ), f"{cache_k.size(0)} != {self.n_layers}"
            assert (
                cache_v.size(0) == self.n_layers
            ), f"{cache_v.size(0)} != {self.n_layers}"

            sp = start_pos.item()
            # self.params.max_seq_len - 1 because of 0 based indexing, and - 1 again because our input seq len is 1 and its added to the cache before accessing the cache
            torch._constrain_as_size(sp, min=0, max=self.params.max_seq_len - 2)
            torch._constrain_as_value(
                cache_k.shape[0],
                max=self.n_layers,
                min=self.n_layers,
            )
            torch._constrain_as_value(
                cache_v.shape[0], min=self.n_layers, max=self.n_layers
            )
            # when KV cache is used, seqlen is most likely 1. We want to slice from the start_pos.
            freqs_cos = self.freqs_cos[sp : sp + seqlen]
            freqs_sin = self.freqs_sin[sp : sp + seqlen]
        else:
            # assert (
            #     start_pos is None and cache_k is None and cache_v is None
            # ), "Caches and start_pos are unused when use_kv_cache is False"
            freqs_cos = self.freqs_cos[:seqlen]
            freqs_sin = self.freqs_sin[:seqlen]

        for index, layer in enumerate(self.layers):
            if self.use_kv_cache:
                if self.params.use_sdpa_with_kv_cache_op:
                    h, updated_cache_k, updated_cache_v = layer(
                        h,
                        freqs_cos,
                        freqs_sin,
                        sp,  # pyre-ignore[61]
                        cache_k,
                        cache_v,
                    )
                else:
                    h, updated_cache_k, updated_cache_v = layer(
                        h,
                        freqs_cos,
                        freqs_sin,
                        sp,  # pyre-ignore[61]
                        cache_k[index],  # pyre-ignore[16]
                        cache_v[index],
                    )
                    cache_k[index] = updated_cache_k  # pyre-ignore[16]
                    cache_v[index] = updated_cache_v

            else:
                h, _, _ = layer(h, freqs_cos, freqs_sin)

        h = self.norm(h)

        logits = self.output(h)
        if self.use_kv_cache:
            return (logits, cache_k, cache_v)  # pyre-ignore
        else:
            # 'None' is not a valid return for export so have to split the return into if else
            return logits

    # For each layer return the sizes of the needed caches
    def get_cache_sizes(self):
        # cache_k and cache_v have the same shape so could pick either here.
        return [self.n_layers, *self.layers[0].attention.kv_cache_sizes]
