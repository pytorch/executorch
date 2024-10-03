#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from torch.nn import functional as F


logger: logging.Logger = logging.getLogger()


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
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

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
    invocation_vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048
    use_layer_norm_op: bool = False
    use_rms_norm_op: bool = False
    hidden_dim: Optional[int] = None


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
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


class Attention(torch.nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = torch.nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = torch.nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = torch.nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = torch.nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        # set large value of -inf (or -32768 with int16) when we want to
        # ignore correspnding values in the mask
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-32768"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # grouped multiquery attention: expand out keys and values
        xk = [
            torch.cat([xk[:, :, i : i + 1, :]] * self.n_rep, dim=2)
            for i in range(xk.size(2))
        ]
        xk = torch.cat(xk, dim=2)

        xv = [
            torch.cat([xv[:, :, i : i + 1, :]] * self.n_rep, dim=2)
            for i in range(xv.size(2))
        ]
        xv = torch.cat(xv, dim=2)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        assert hasattr(self, "mask")
        scores = (
            scores + self.mask[:, :, :seqlen, :seqlen]
        )  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        output = self.wo(output)
        return output


class FeedForward(torch.nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int):
        super().__init__()
        self.w1 = torch.nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = torch.nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = torch.nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        x = F.silu(self.w1(x)) * self.w3(x)
        x = self.w2(x)
        return x


class TransformerBlock(torch.nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        if args.hidden_dim is None:
            hidden_dim = 4 * args.dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = args.multiple_of * (
                (hidden_dim + args.multiple_of - 1) // args.multiple_of
            )
        else:
            hidden_dim = args.hidden_dim
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=hidden_dim,
            multiple_of=args.multiple_of,
        )
        self.layer_id = layer_id
        if args.use_layer_norm_op:
            self.attention_norm = torch.nn.LayerNorm(args.dim, eps=args.norm_eps)
            self.ffn_norm = torch.nn.LayerNorm(args.dim, eps=args.norm_eps)
        elif args.use_rms_norm_op:
            self.attention_norm = torch.nn.RMSNorm(args.dim, eps=args.norm_eps)
            self.ffn_norm = torch.nn.RMSNorm(args.dim, eps=args.norm_eps)
        else:
            self.attention_norm = torch.nn.RMSNorm(args.dim, eps=args.norm_eps)
            self.ffn_norm = torch.nn.RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class LastTimeStepPool(torch.nn.Module):
    def forward(self, logits: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        bsz, _, dim = logits.shape
        idx = seq_lens.unsqueeze(1).expand(bsz, dim).unsqueeze(1)
        return logits.gather(1, idx - 1).squeeze(1)


class Transformer(torch.nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = torch.nn.Embedding(params.vocab_size, params.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        if params.use_layer_norm_op:
            self.norm = torch.nn.LayerNorm(params.dim, eps=params.norm_eps)
        elif params.use_rms_norm_op:
            self.norm = torch.nn.RMSNorm(params.dim, eps=params.norm_eps)
        else:
            self.norm = torch.nn.RMSNorm(params.dim, eps=params.norm_eps)
        self.out = torch.nn.Linear(params.dim, params.vocab_size, bias=False)

        freqs_cos, freqs_sin = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)

        h = self.norm(h)

        invocation_logits = self.out(h)

        return invocation_logits
