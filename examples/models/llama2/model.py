# @lint-ignore-every LICENSELINT
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Llama 2 is licensed under the LLAMA 2 Community License,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

# Please refer to README.md in the same folder for more information.


import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from torch import nn

from ..model_base import EagerModelBase


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
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048
    moe: bool = False  # True to enable the MoE (Mixture of Experts)
    num_experts: int = 8  # Number of experts
    num_activated_experts: int = 2  # Number of experts to activate


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


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        # args.dim = 4096, args.n_heads = 32, self.head_dim = 4096 / 32 = 125
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
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
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        assert hasattr(self, "mask")
        mask = self.mask[:, :, :seqlen, :seqlen]
        output = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=mask, dropout_p=0.0
        )

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        output = self.wo(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ConditionalFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = config.multiple_of * (
            (hidden_dim + config.multiple_of - 1) // config.multiple_of
        )
        self.w1 = nn.Parameter(torch.randn(config.num_experts, hidden_dim, config.dim))
        self.w2 = nn.Parameter(torch.randn(config.num_experts, hidden_dim, config.dim))
        self.w3 = nn.Parameter(torch.randn(config.num_experts, hidden_dim, config.dim))
        self.num_experts = config.num_experts
        self.dim = config.dim

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
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        if args.moe:
            self.block_sparse_moe = MOEFeedForward(args)
        else:
            self.feed_forward = FeedForward(
                dim=args.dim,
                hidden_dim=4 * args.dim,
                multiple_of=args.multiple_of,
            )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):  # x: 1xN
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
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
        # h = self.layers[0](h, freqs_cos, freqs_sin) # myuan: hack one layer for debug

        h = self.norm(h)

        logits = self.output(h)
        return logits


class Llama2Model(EagerModelBase):
    def __init__(self, **kwargs):
        import pkg_resources

        # default path to the resource file
        # It currently supports 3 ways of specifying the checkpoint location:
        # 1. Using default path locates in examples/models/llama2/params
        # 2. Passing in the checkpoint path and params via kwargs
        # 3. Using the path from pkg_resources, only works with buck2
        try:
            # The 3rd way, if we can import this path, we are running with buck2, all resources can be accessed with pkg_resources.resource_filename
            # pyre-ignore
            from executorch.examples.models.llama2 import params

            ckpt_dir = Path(
                pkg_resources.resource_filename(
                    "executorch.examples.models.llama2", "params"
                )
            )
        except:
            # The 1st way
            ckpt_dir = Path(__file__).absolute().parent / "params"

        checkpoint_path = (
            kwargs["checkpoint"]
            if "checkpoint" in kwargs
            else ckpt_dir / "demo_rand_params.pth"
        )

        params_path = (
            kwargs["params"] if "params" in kwargs else ckpt_dir / "demo_config.json"
        )

        # The example is using a dummy small model with random weights for demo purpose only.
        # Follow the instruction in https://github.com/facebookresearch/llama to download the model
        device = "cpu"
        # flake8: noqa: TOR102
        checkpoint = torch.load(checkpoint_path, map_location=device)
        with open(params_path, "r") as f:
            params = json.loads(f.read())
        max_seq_len = 128
        max_batch_size = 1
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        self.model_ = Transformer(model_args)
        self.model_.load_state_dict(
            checkpoint, strict=False
        )  # self.model_ = Transformer(gptconf)

    # @staticmethod
    def get_eager_model(self):
        return self.model_

    @staticmethod
    def get_example_inputs():
        return (torch.tensor([[1, 2]], dtype=torch.long),)
