# @lint-ignore-every LICENSELINT
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

# Please refer to README.md in the same folder for more information.

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from executorch.examples.models.llama.norm import RMSNorm

from executorch.examples.models.llama.rope import (
    hf_apply_rotary_emb,
    hf_precompute_freqs_cis,
    precompute_freqs_cis,
    RotaryEmbedding,
)

from torch import nn

logger = logging.getLogger(__name__)


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def silu_approx(x):
    x = x.clamp(-3, 3)
    x2 = x * x
    x4 = x2 * x2
    x6 = x4 * x2
    res = 0.0017 + 0.5 * x + 0.2423  * x2 -0.0153 * x4 + 0.00057 * x6
    return res

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
    high_freq_factor: int = 4

    bos_idx: int = 1
    eos_idx: int = 3
    bos_count: int = -1  # i.e., a single EOS is used as BOS
    eos_count: int = 2

    quantization_args: Optional[dict] = None
    lora_args: Optional[dict] = None

    use_cache_list: bool = True

    use_kv_cache: bool = False
    enable_dynamic_shape: bool = False

    use_qk_norm: bool = False
    qk_norm_before_rope: bool = False

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

def rms_norm_fp16_stable(x, eps=1e-5, min_scale=1e-3):
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = amax.clamp(min=min_scale)
    x_scaled = x / scale

    var = torch.square(x_scaled).mean(dim=-1, keepdim=True)
    rms = torch.sqrt(var + eps)
    y = x_scaled / rms
    return y

class CoreMLRMSNorm(torch.nn.Module):
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
        # CoreML ignores casts to FP32, so existing implementation of RMSNorm was not stable
        # We instead use (x * sqrt(n)) / norm(x, dim=-1)
        # Using torch.norm and preserving this op in CoreML improves stability
        # Note, we ignore eps, but could add it by using torch.norm(torch.concat(x, sqrt(n*eps))) in the denominator
        # In future, we want to add CoreML support for the functional RMSNorm op
        # We have yet to do large scale evaluations on the numeric stability of this solution, but note that
        # it appears better than what exists currently (removing FP32 casts and using FP16)
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        rms_norm_eps0 = (
            x
            * (torch.sqrt(torch.tensor(self.dim, dtype=x.dtype)) / norm)
            # * torch.reciprocal(torch.linalg.vector_norm(x, dim=-1, keepdim=True))
        )
        return rms_norm_eps0

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x)
        return output * self.weight

_RMS_NORM = CoreMLRMSNorm
_DECOMPOSE_SDPA = True
_USE_SOFTMAX = True
_USE_SILU_APPROX = False

class Rope(torch.nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        if self.params.use_hf_rope:
            self.precompute_freqs_cis = partial(
                hf_precompute_freqs_cis,
                partial_rotary_factor=self.params.partial_rotary_factor,
            )
        else:
            self.precompute_freqs_cis = partial(
                precompute_freqs_cis,
                use_scaled=self.params.use_scaled_rope,
                scale_factor=self.params.rope_scale_factor,
                high_freq_factor=self.params.high_freq_factor,
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
        t1 = self.w1(x)
        if _USE_SILU_APPROX:
            t1 = silu_approx(t1)
        else:
            t1 = F.silu(t1)
        t2 = self.w3(x)
        out = t1 * t2
        out = self.w2(out)
        return out

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

        self.use_qk_norm = args.use_qk_norm
        self.qk_norm_before_rope = args.qk_norm_before_rope
        if self.use_qk_norm:
            q_norm_dim = self.head_dim
            k_norm_dim = self.head_dim
            self.q_norm_fn = _RMS_NORM(q_norm_dim, eps=args.norm_eps)
            self.k_norm_fn = _RMS_NORM(k_norm_dim, eps=args.norm_eps)

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

        if self.use_qk_norm and not self.qk_norm_before_rope:
            q = self.q_norm_fn(q)
            k = self.k_norm_fn(k)

        new_k = k
        new_v = v

        k = torch.concat([k_cache, k], dim=2)
        v = torch.concat([v_cache, v], dim=2)

        # TODO: I'm pretty sure the MB version of SDPA does not require this repeat_interleave, 
        # grouped multiquery attention: expand out keys and values
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        if not _DECOMPOSE_SDPA:
            output = torch.ops.aten.scaled_dot_product_attention.default(
                q, k, v, attn_mask=attn_mask
            )
        else:

            # ------------------------------
            # Manual SDPA: matmuls + softmax
            # q: (B, H, T_q, D)
            # k: (B, H, T_k, D)
            # v: (B, H, T_k, D)
            # attn_mask: broadcastable to (B, H, T_q, T_k)
            # ------------------------------
            d = q.size(-1)
            # (B, H, T_q, T_k)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)

            if attn_mask is not None:
                # attn_mask is already used this way with SDPA, keep same semantics:
                # 0.0 for allowed, -inf for disallowed, added to scores.
                scores = scores + attn_mask

            if _USE_SOFTMAX:
                # (B, H, T_q, T_k)
                attn_weights = torch.softmax(scores, dim=-1)
            else:
                scores = scores.clamp(min=-60.0, max=60.0)
                scores_max, _ = scores.max(dim=-1, keepdim=True)          # (B, H, T_q, 1)
                scores_exp = torch.exp(scores - scores_max)
                attn_weights = scores_exp / scores_exp.sum(dim=-1, keepdim=True)

            # (B, H, T_q, D)
            output = torch.matmul(attn_weights, v)

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
        self.attention_norm = _RMS_NORM(args.dim, eps=args.norm_eps)
        self.ffn_norm = _RMS_NORM(args.dim, eps=args.norm_eps)

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
        tmp = self.feed_forward(self.ffn_norm(h))
        out = h + tmp
        return out, new_k, new_v



class AttentionBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, rope: Rope):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.head_dim
        self.attention = Attention(args, layer_id, rope)
        self.attention_norm = _RMS_NORM(args.dim, eps=args.norm_eps)

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
        return h, new_k, new_v


class FeedForwardBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, rope: Rope):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.head_dim
        if args.moe:
            self.block_sparse_moe = MOEFeedForward(args)
        else:
            self.feed_forward = FeedForward(args)
        self.ffn_norm = _RMS_NORM(args.dim, eps=args.norm_eps)

    def forward(
        self,
        h,
    ):  # x: 1xN
        tmp = self.feed_forward(self.ffn_norm(h))
        out = h + tmp
        return out



class InputBlock(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.rope = Rope(params)
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
    
    def forward(self, tokens: torch.LongTensor, input_pos: torch.LongTensor):
        h = self.tok_embeddings(tokens)
        seqlen = h.shape[1]
        freqs_cos, freqs_sin = self.rope.get_freqs(input_pos, seqlen)
        return h, freqs_cos, freqs_sin 

class OutputBlock(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.generate_full_logits = params.generate_full_logits
        self.norm = _RMS_NORM(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
    
    def forward(self, h, input_length: torch.LongTensor):
        if not self.generate_full_logits:
            # Only the last logit is used for the new generated token
            h = h[:, input_length - 1, :].squeeze(1)
        h = self.norm(h)
        logits = self.output(h)
        return logits


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
        self.norm = _RMS_NORM(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.generate_full_logits = params.generate_full_logits
        self.max_seq_len = params.max_seq_len
        self.input_prune_map = params.input_prune_map
        self.output_prune_map = params.output_prune_map
        self.use_cache_list = params.use_cache_list

    def forward(
        self,
        tokens: torch.LongTensor,  # tokens
        input_pos: torch.LongTensor,
        input_length: torch.LongTensor,  # input_length
        k_caches: List[torch.FloatTensor],
        v_caches: List[torch.FloatTensor],
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
                k_caches[i] if self.use_cache_list else k_caches[i, :, :, :, :],
                v_caches[i] if self.use_cache_list else v_caches[i, :, :, :, :],
                attn_mask,
            )
            k_out.append(new_k)
            v_out.append(new_v)

        if not self.generate_full_logits:
            # Only the last logit is used for the new generated token
            h = h[:, input_length - 1, :].squeeze(1)

        h = self.norm(h)

        logits = self.output(h)

        if not self.use_cache_list:
            k_out = torch.stack(k_out, dim=0)
            v_out = torch.stack(v_out, dim=0)
        return logits, k_out, v_out  # pyre-ignore[7]


def load_model(checkpoint_path, params_path, max_seq_length, use_cache_list):
    import json

    with open(params_path, "r") as f:
        params = json.loads(f.read())

    args = ModelArgs(
        max_seq_len=max_seq_length,
        generate_full_logits=False,
        use_cache_list=use_cache_list,
        **params,
    )

    with torch.device("meta"):
        model = Transformer(args)

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=True
    )
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]

    missing, unexpected = model.load_state_dict(
        checkpoint,
        strict=False,
        assign=True,
    )
    print("Missing keys: ", missing)
    print("Unexpected keys: ", unexpected)

    return model

def load_model_in_pieces_ITO(checkpoint_path, params_path, max_seq_length, seq_length, float_dtype: torch.dtype = torch.float16):
    """
    Loads model in pieces: input, [repeating transformer], output
    """
    import json

    with open(params_path, "r") as f:
        params = json.loads(f.read())

    args = ModelArgs(
        max_seq_len=max_seq_length,
        generate_full_logits=False,
        **params,
    )

    with torch.device("meta"):
        rope = Rope(args)
        in_block = InputBlock(args)
        out_block = OutputBlock(args)
        transformer_blocks = []
        for layer_id in range(args.n_layers):
            transformer_blocks.append(TransformerBlock(layer_id, args, rope))
    
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=True
    )
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]

    for model in [in_block, out_block]:
        missing, unexpected = model.load_state_dict(
            checkpoint,
            strict=False,
            assign=True,
        )
        assert len(missing) == 0
        
    for i, model in enumerate(transformer_blocks):
        layer_prefix = f"layers.{i}."
        layer_checkpoint = {k[len(layer_prefix):]:v for k, v in checkpoint.items() if k.startswith(layer_prefix)}
        missing, unexpected = model.load_state_dict(
            layer_checkpoint,
            strict=False,
            assign=True,
        )
        assert len(missing) == 0

    cache_shape = (1, args.n_kv_heads, max_seq_length - seq_length, args.head_dim)
    attn_mask_shape = (seq_length, max_seq_length)
    freqs_shape = (seq_length, args.head_dim // 2)
    h_shape = (1, seq_length, args.dim)

    example_inputs = [
        (torch.zeros(1, seq_length, dtype=torch.int64), torch.tensor([0], dtype=torch.long),), # InputBlock
        ( # TransformerBlock
            torch.zeros(h_shape, dtype=float_dtype), # h
            torch.zeros(freqs_shape, dtype=float_dtype), # freqs_cos
            torch.zeros(freqs_shape, dtype=float_dtype), # freqs_sin
            torch.zeros(cache_shape, dtype=float_dtype), # k_cache
            torch.zeros(cache_shape, dtype=float_dtype), # v_cache
            torch.zeros(attn_mask_shape, dtype=float_dtype),
        ),
        ( # OutputBlock
            torch.zeros(h_shape, dtype=float_dtype), # h
            torch.tensor([0], dtype=torch.long), # input_length
        ),
    ]

    models = [in_block] + transformer_blocks + [out_block]
    for i in range(len(models)):
        models[i] = models[i].to(float_dtype)

    return models, example_inputs

def load_model_in_pieces_IAFO(checkpoint_path, params_path, max_seq_length, seq_length, float_dtype: torch.dtype = torch.float16):
    """
    Loads model in pieces: input, [repeating attention, feedforward], output
    """
    import json

    with open(params_path, "r") as f:
        params = json.loads(f.read())

    args = ModelArgs(
        max_seq_len=max_seq_length,
        generate_full_logits=False,
        **params,
    )

    with torch.device("meta"):
        rope = Rope(args)
        in_block = InputBlock(args)
        out_block = OutputBlock(args)
        transformer_blocks = []
        for layer_id in range(args.n_layers):
            transformer_blocks.append(AttentionBlock(layer_id, args, rope))
            transformer_blocks.append(FeedForwardBlock(layer_id, args, rope))

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=True
    )
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]


    for model in [in_block, out_block]:
        missing, unexpected = model.load_state_dict(
            checkpoint,
            strict=False,
            assign=True,
        )
        assert len(missing) == 0
        

    for i in range(len(transformer_blocks) // 2):
        layer_prefix = f"layers.{i}."
        layer_checkpoint = {k[len(layer_prefix):]:v for k, v in checkpoint.items() if k.startswith(layer_prefix)}
        missing, unexpected = transformer_blocks[2*i].load_state_dict(
            layer_checkpoint,
            strict=False,
            assign=True,
        )
        assert len(missing) == 0
        missing, unexpected = transformer_blocks[2*i+1].load_state_dict(
            layer_checkpoint,
            strict=False,
            assign=True,
        )
        assert len(missing) == 0

    cache_shape = (1, args.n_kv_heads, max_seq_length - seq_length, args.head_dim)
    attn_mask_shape = (seq_length, max_seq_length)
    freqs_shape = (seq_length, args.head_dim // 2)
    h_shape = (1, seq_length, args.dim)

    example_inputs = [
        (torch.zeros(1, seq_length, dtype=torch.int64), torch.tensor([0], dtype=torch.long),), # InputBlock
        ( # AttentionBlock
            torch.zeros(h_shape, dtype=float_dtype), # h
            torch.zeros(freqs_shape, dtype=float_dtype), # freqs_cos
            torch.zeros(freqs_shape, dtype=float_dtype), # freqs_sin
            torch.zeros(cache_shape, dtype=float_dtype), # k_cache
            torch.zeros(cache_shape, dtype=float_dtype), # v_cache
            torch.zeros(attn_mask_shape, dtype=float_dtype),
        ),
        ( # FeedForwardBlock
            torch.zeros(h_shape, dtype=float_dtype), # h
        ),
        ( # OutputBlock
            torch.zeros(h_shape, dtype=float_dtype), # h
            torch.tensor([0], dtype=torch.long), # input_length
        ),
    ]

    models = [in_block] + transformer_blocks + [out_block]
    for i in range(len(models)):
        models[i] = models[i].to(float_dtype)

    return models, example_inputs


class InputManager:
    class NGramCache:
        def __init__(self, max_size: int):
            self.cache = deque()
            self.max_size = max_size

        def add(self, ngram: List[int]):
            if ngram in self.cache:
                return
            if len(self.cache) == self.max_size:
                self.cache.popleft()
            self.cache.append(ngram)

        def __iter__(self):
            return iter(self.cache)

        def __str__(self):
            return str(self.cache)

    def __init__(
        self,
        n_layers: int,
        max_batch_size: int,
        n_kv_heads: int,
        max_seq_length: int,
        head_dim: int,
        use_cache_list: bool,
        seq_length: int,
        dtype=torch.float16,
        minus_infinity=-torch.inf,
        cache_size=None,
        lookahead_enabled: bool = False,
    ):
        if cache_size is None:
            cache_size = max_seq_length - seq_length
        self.cache_size = cache_size
        assert self.cache_size + seq_length <= max_seq_length

        self.n_layers = n_layers
        self.max_batch_size = max_batch_size
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.seq_length = seq_length
        self.use_cache_list = use_cache_list
        self.lookahead_enabled = lookahead_enabled
        self.minus_infinity = minus_infinity

        if self.use_cache_list:
            self.k_caches = [
                torch.zeros(self.get_cache_shape(self.cache_size)).to(dtype)
                for _ in range(self.n_layers)
            ]
            self.v_caches = [
                torch.zeros(self.get_cache_shape(self.cache_size)).to(dtype)
                for _ in range(self.n_layers)
            ]
        else:
            self.k_caches = torch.zeros(self.get_cache_shape(self.cache_size)).to(dtype)
            self.v_caches = torch.zeros(self.get_cache_shape(self.cache_size)).to(dtype)

        attn_cache = minus_infinity * torch.ones(
            seq_length, self.cache_size
        )  # attn for past tokens
        attn_seq = torch.triu(
            minus_infinity * torch.ones(self.seq_length, self.seq_length), diagonal=1
        )  # attn for current tokens
        self.attn_mask = torch.concat([attn_cache, attn_seq], dim=-1).to(dtype)
        assert self.attn_mask.shape == (
            self.seq_length,
            self.cache_size + self.seq_length,
        )

        self.input_pos = 0
        self.cache_pos = 0

    def get_cache_shape(self, length):
        if self.use_cache_list:
            return (
                self.max_batch_size,
                self.n_kv_heads,
                length,
                self.head_dim,
            )
        return (
            self.n_layers,
            self.max_batch_size,
            self.n_kv_heads,
            length,
            self.head_dim,
        )

    def _update_cache(self, start, length, new_k_caches, new_v_caches):
        """
        Copies new cache data from start to start + length to cache
        """
        assert self.cache_pos + length <= self.cache_size
        assert start + length <= self.seq_length

        if self.use_cache_list:
            for i in range(self.n_layers):
                assert new_k_caches[i].shape == self.get_cache_shape(self.seq_length)
                assert new_v_caches[i].shape == self.get_cache_shape(self.seq_length)

                self.k_caches[i][
                    :, :, (self.cache_pos) : (self.cache_pos + length), :
                ] = new_k_caches[i][:, :, start : (start + length), :]
                self.v_caches[i][
                    :, :, (self.cache_pos) : (self.cache_pos + length), :
                ] = new_v_caches[i][:, :, start : (start + length), :]
        else:
            assert new_k_caches.shape == self.get_cache_shape(self.seq_length)
            assert new_v_caches.shape == self.get_cache_shape(self.seq_length)
            self.k_caches[:, :, :, (self.cache_pos) : (self.cache_pos + length), :] = (
                new_k_caches[:, :, :, start : (start + length), :]
            )
            self.v_caches[:, :, :, (self.cache_pos) : (self.cache_pos + length), :] = (
                new_v_caches[:, :, :, start : (start + length), :]
            )

        self.cache_pos += length
        if self.cache_pos == self.cache_size:
            self.cache_pos = 0

    def update(self, input_length, new_k_caches, new_v_caches, update_pos=0):
        # Copy as much new cache data into cache as possible without wrapping
        amount_to_copy = min(input_length, self.cache_size - self.cache_pos)
        self._update_cache(update_pos, amount_to_copy, new_k_caches, new_v_caches)
        if self.input_pos <= self.cache_size:
            self.attn_mask[:, (self.input_pos) : (self.input_pos + amount_to_copy)] = (
                0.0
            )

        # Copy remainder (cache is now wrapped around and has more room)
        # Attention mask needs no further updates.  Attention is paid to the whole cache
        remaining_to_copy = min(
            input_length - amount_to_copy, self.cache_size - self.cache_pos
        )
        if remaining_to_copy > 0:
            self._update_cache(
                update_pos + amount_to_copy,
                remaining_to_copy,
                new_k_caches,
                new_v_caches,
            )

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
                dim=-1,
            ).reshape(1, -1),
            # input_pos
            torch.tensor([self.input_pos], dtype=torch.long),
            # input_length
            torch.tensor([input_length], dtype=torch.long),
            # k_cache
            self.k_caches,
            # v_cache
            self.v_caches,
            # attn_mask
            self.attn_mask,
        )

    def get_inputs_and_remaining_tokens(self, tokens: List[int]):
        processed_tokens = min(self.seq_length, len(tokens))
        return (
            self.get_inputs(tokens[0:processed_tokens]),
            tokens[processed_tokens:],
        )

    def _get_lookahead_decoding_mask(
        self, ngram_size: int, window_size: int, n_verifications: int
    ) -> torch.Tensor:
        mask = torch.full((self.seq_length, self.seq_length), self.minus_infinity)
        mask[0][0] = 0.0

        lookahead_submask = torch.triu(
            torch.full((window_size, window_size), self.minus_infinity),
            diagonal=1,
        )
        for i in range(ngram_size - 1):
            offset = window_size * i
            mask[offset : offset + window_size, :window_size] = lookahead_submask
            for j in range(1, i + 1):
                mask[
                    offset : offset + window_size,
                    window_size * j : window_size * (j + 1),
                ].fill_diagonal_(0.0)

        verification_offset = max(window_size * (ngram_size - 1), 1)
        verification_submask = torch.triu(
            torch.full((ngram_size - 1, ngram_size - 1), self.minus_infinity),
            diagonal=1,
        )
        for i in range(n_verifications):
            mask[
                verification_offset
                + i * (ngram_size - 1) : verification_offset
                + (i + 1) * (ngram_size - 1),
                verification_offset
                + i * (ngram_size - 1) : verification_offset
                + (i + 1) * (ngram_size - 1),
            ] = verification_submask
        mask[verification_offset:, :1] = 0.0

        return mask

    def _get_lookahead_position_offsets(
        self, ngram_size: int, window_size: int, n_verifications: int
    ) -> torch.Tensor:
        pos_offsets = torch.zeros(self.seq_length, dtype=torch.int32)
        idx = 0
        if window_size > 0:
            for i in range(ngram_size - 1):
                for j in range(window_size):
                    pos_offsets[idx] = i + j
                    idx += 1
        else:
            pos_offsets[0] = 0
            idx += 1

        # Verification branches: [1, 2, ..., ngram_size - 1].
        for _ in range(n_verifications):
            for j in range(1, ngram_size):
                pos_offsets[idx] = j
                idx += 1

        return pos_offsets

    def _validate_lookahead_config(
        self, ngram_size: int, window_size: int, n_verifications: int
    ) -> None:
        """
        Validate the lookahead decoding configuration.
        """
        if not self.lookahead_enabled:
            raise RuntimeError("Lookahead decoding is not enabled")

        if (ngram_size - 1) * (window_size + n_verifications) > self.seq_length:
            raise RuntimeError(
                f"Lookahead decoding configuration not compatible with seq_length {self.seq_length}. "
                f"Required: {(ngram_size - 1) * (window_size + n_verifications)}"
            )

    def _setup_lookahead_mask(
        self, ngram_size: int, window_size: int, n_verifications: int
    ) -> None:
        """
        Set up the attention mask for lookahead decoding and log debug information.
        """
        self.attn_mask[:, self.cache_size :] = self._get_lookahead_decoding_mask(
            ngram_size, window_size, n_verifications
        )
        logger.debug("Lookahead decoding mask: ")
        for i in range(self.seq_length):
            logger.debug(
                " ".join(
                    ("X" if x == 0.0 else " ")
                    for x in self.attn_mask[i][self.cache_size :]
                )
            )

    def _populate_verification_branches(
        self, x: List[int], cache, verification_offset: int, ngram_size: int
    ) -> None:
        """
        Populate verification branches with tokens from the n-gram cache.
        """
        for i, ngram in enumerate(cache):
            for j, token in enumerate(ngram):
                x[verification_offset + i * (ngram_size - 1) + j] = token

    def _collect_ngrams(
        self,
        x: List[int],
        y: List[int],
        ngram_caches: Dict[int, "InputManager.NGramCache"],
        window_size: int,
        ngram_size: int,
    ) -> None:
        """
        Collect new n-grams from the current state and predictions.
        """
        for i in range(window_size):
            key = x[i]
            suffix = []
            for j in range(1, ngram_size - 1):
                suffix.append(x[i + j * window_size])
            suffix.append(y[i + window_size * (ngram_size - 2)])
            ngram_caches[key].add(suffix)

    def _find_longest_match(
        self,
        x: List[int],
        y: List[int],
        verification_offset: int,
        n_verifications: int,
        ngram_size: int,
    ) -> Tuple[List[int], Optional[int]]:
        """
        Find the longest matching sequence from verification branches.
        Returns the matched tokens and the branch index.
        """
        longest_match = []
        matched_branch = None

        for i in range(n_verifications):
            match = [y[0]]
            j = 0
            while (
                j < ngram_size - 1
                and x[verification_offset + (ngram_size - 1) * i + j] == match[-1]
            ):
                match.append(y[verification_offset + (ngram_size - 1) * i + j])
                j += 1
            if len(match) - 1 > len(longest_match):
                longest_match = match[1:]
                matched_branch = i

        return longest_match, matched_branch

    def _update_lookahead_branches(
        self, x: List[int], y: List[int], ngram_size: int, window_size: int
    ) -> None:
        """
        Update the lookahead branches with new predictions.
        """
        # Shift window contents up
        for i in range(ngram_size - 2):
            for j in range(window_size):
                x[window_size * i + j] = x[window_size * (i + 1) + j]

        # Fill the last window with new predictions
        for j in range(window_size):
            x[window_size * (ngram_size - 2) + j] = y[
                window_size * (ngram_size - 2) + j
            ]

    def lookahead_decode(
        self,
        model,
        init_token: int,
        n: int,
        ngram_size: int,
        window_size: int,
        n_verifications: int,
        stop_tokens: Optional[List[int]] = None,
        ngram_caches: Optional[Dict[int, "InputManager.NGramCache"]] = None,
    ) -> List[int]:
        # Validate configuration
        self._validate_lookahead_config(ngram_size, window_size, n_verifications)

        # Setup attention mask and position offsets
        self._setup_lookahead_mask(ngram_size, window_size, n_verifications)
        offsets = self._get_lookahead_position_offsets(
            ngram_size, window_size, n_verifications
        )

        # Initialize state
        stop_tokens = stop_tokens or []
        verification_offset = window_size * (ngram_size - 1)
        if ngram_caches is None:
            ngram_caches = defaultdict(lambda: InputManager.NGramCache(n_verifications))

        new_tokens = [init_token]
        x = [init_token] * self.seq_length
        inference_count = 0

        # Main decoding loop
        while len(new_tokens) < n + 1:
            # Populate verification branches
            cache = ngram_caches[x[0]]
            self._populate_verification_branches(
                x, cache, verification_offset, ngram_size
            )

            # Run model inference
            logits, new_k, new_v = model(
                tokens=torch.tensor([x], dtype=torch.int64),
                input_pos=torch.tensor([self.input_pos], dtype=torch.long),
                k_caches=self.k_caches,
                v_caches=self.v_caches,
                attn_mask=self.attn_mask,
                input_len=torch.tensor([len(x)], dtype=torch.long),
                rope_indices=self.input_pos + offsets,
            )
            inference_count += 1

            # Process model output (greedy selection)
            y = logits[0].argmax(dim=-1).tolist()
            new_tokens.append(y[0])
            logger.debug(f"{self.input_pos}: x = {x[0]}, y = {y[0]}")
            if new_tokens[-1] in stop_tokens:
                break

            # Collect new n-grams
            self._collect_ngrams(x, y, ngram_caches, window_size, ngram_size)

            # Find longest match from verification branches
            longest_match, matched_branch = self._find_longest_match(
                x, y, verification_offset, n_verifications, ngram_size
            )

            # Process match results
            if matched_branch is not None:
                logger.debug(
                    f"Matched {len(longest_match)} additional tokens from n-grams: {longest_match}"
                )
                # Truncate at stop token if present
                for stop in stop_tokens:
                    if stop in longest_match:
                        longest_match = longest_match[: longest_match.index(stop) + 1]

                new_tokens.extend(longest_match)
                branch_offset = verification_offset + (ngram_size - 1) * matched_branch
                self.update(
                    input_length=len(longest_match),
                    new_k_caches=new_k,
                    new_v_caches=new_v,
                    update_pos=branch_offset,
                )
            else:
                self.update(input_length=1, new_k_caches=new_k, new_v_caches=new_v)

            # Update lookahead branches
            self._update_lookahead_branches(x, y, ngram_size, window_size)

            # Update first token and check for stop condition
            x[0] = new_tokens[-1]
            if new_tokens[-1] in stop_tokens:
                break

        logger.info(
            f"Generated {len(new_tokens) - 1} tokens with {inference_count} inference(s)."
        )
        return new_tokens
