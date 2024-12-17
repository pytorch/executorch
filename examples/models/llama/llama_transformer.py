# @lint-ignore-every LICENSELINT
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Llama 2 is licensed under the LLAMA 2 Community License,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

# Please refer to README.md in the same folder for more information.

from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from executorch.examples.models.llama.rope import (
    hf_apply_rotary_emb,
    hf_precompute_freqs_cis,
    precompute_freqs_cis,
    RotaryEmbedding,
)

from torch import nn

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
    head_dim: Optional[int] = None  # Optional customized head_dim
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048
    moe: bool = False  # True to enable the MoE (Mixture of Experts)
    num_experts: int = 8  # Number of experts
    num_activated_experts: int = 2  # Number of experts to activate
    use_kv_cache: bool = False  # Use key/value cache
    prefill_return_kv: bool = False  # Return kv cache for prefill
    decode_kv_cache_as_io: bool = False # Decode uses KV caches as IO
    use_additive_kv_cache_update: bool = False # Additive KV cache update
    use_sdpa_with_kv_cache_op: bool = (
        False  # Use custom sdpa op that updates kv cache in-place
    )
    # Generate logits for all inputs. When it's True, it would take big memory usage
    # at runtime. Enable it only necessary (e.g., use perplexity tools that requires
    # logits for all input tokens.)
    generate_full_logits: bool = False
    enable_dynamic_shape: bool = False  # export model with dynamic shape support
    # A dictionary mapping from pruned token-id to original token-id
    input_prune_map: Optional[Dict[int, int]] = None
    # A dictionary mapping from pruned token-id to original token-id
    output_prune_map: Optional[Dict[int, int]] = None
    use_hf_rope: bool = False  # Use HuggingFace's RoPE implementation
    rope_theta: Optional[float] = (
        None  # The official name to override self.rope_freq_base.
    )
    rope_freq_base: float = 10000.0  # The base frequency for RoPE. Keep it for BC.
    use_scaled_rope: bool = False  # Use scaled RoPE, introduced in llama3.1.
    rope_scale_factor: int = 8
    # Additional Model Metadata needed at runtime
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
                precompute_freqs_cis,
                use_scaled=self.params.use_scaled_rope,
                scale_factor=self.params.rope_scale_factor,
            )
        freqs_cos, freqs_sin = self.precompute_freqs_cis(
            self.params.head_dim,
            (
                self.params.max_seq_len  # Normal llama2.
                if self.params.ffn_dim_multiplier is None
                else self.params.max_seq_len * 2  # Sharded checkpoint.
            ),
            self.params.rope_freq_base,
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
        if self.params.use_kv_cache:
            assert (
                input_pos is not None
            ), "input_pos must be provided when use_kv_cache is True"

            if self.params.enable_dynamic_shape:
                # when KV cache is used, seqlen is most likely 1. We want to slice from the start_pos.
                input_pos_item = input_pos[-1].item()
                torch._check_is_size(input_pos_item)
                torch._check(input_pos_item < self.params.max_seq_len)
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
        self.is_transposed = transpose_cache
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
            start_pos = input_pos[0].item()
            torch._check_is_size(start_pos)
            torch._check(start_pos < self.max_seq_length)
            dim_to_slice = 2 if self.transpose_cache else 1
            seq_length = k_val.size(dim_to_slice)
            # Replace the entry in the cache for this token
            # The following lines are equivalent to:
            # cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            # cache_v[:bsz, start_pos : start_pos + seqlen] = xv
            # when dim_to_slice is 1
            # We use .narrow() here to make the compiler happy
            # pyre-ignore: Incompatible parameter type [6]
            narrowed_k = self.k_cache.narrow(dim_to_slice, start_pos, seq_length)
            # pyre-ignore: Incompatible parameter type [6]
            narrowed_v = self.v_cache.narrow(dim_to_slice, start_pos, seq_length)

            narrowed_k.copy_(k_val)
            narrowed_v.copy_(v_val)
            return self.k_cache, self.v_cache
        else:
            k_out = self.k_cache
            v_out = self.v_cache
            if self.transpose_cache:
                k_out[:, :, input_pos] = k_val
                v_out[:, :, input_pos] = v_val
            else:
                k_out[:, input_pos] = k_val
                v_out[:, input_pos] = v_val

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
        q: torch.Tensor,  # Already have rotary embeddings. (bs, seqlen, n_local_heads, head_dim)
        k: torch.Tensor,  # Already have rotary embeddings. (bs, seqlen, n_local_kv_heads, head_dim)
        v: torch.Tensor,  # (bs, seqlen, n_local_kv_heads, head_dim)
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
    def __init__(self, args: ModelArgs, layer_id: int, rope: Rope):
        super().__init__()
        self.use_kv_cache = args.use_kv_cache
        self.decode_kv_cache_as_io = args.decode_kv_cache_as_io
        self.use_additive_kv_cache_update = args.use_additive_kv_cache_update
        self.return_kv_values = (args.prefill_return_kv or args.decode_kv_cache_as_io)
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

        causal_mask = torch.tril(
            torch.ones(
                self.max_seq_len,
                self.max_seq_len,
                dtype=torch.bool,
                device="cpu",
            )
        )
        self.register_buffer("mask", causal_mask, persistent=False)

        if self.use_kv_cache and not self.decode_kv_cache_as_io:
            self.kv_cache = KVCache(
                args.max_batch_size,
                args.max_seq_len,
                self.n_kv_heads,
                self.head_dim,
                not args.use_sdpa_with_kv_cache_op,  # if we are using the custom op don't transpose the cache. Expect untransposed q k v
                args.enable_dynamic_shape,
            )
            self.SDPA = SDPA(
                kv_cache=self.kv_cache,
                dim=self.n_local_heads * self.head_dim,
                head_dim=self.head_dim,
                n_rep=self.n_rep,
                max_seq_len=self.max_seq_len,
                enable_dynamic_shape=args.enable_dynamic_shape,
            )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
        cache_pos_mask: Optional[torch.Tensor] = None,
    ):
        if self.decode_kv_cache_as_io:
            assert self.use_kv_cache
            assert k_cache is not None
            assert v_cache is not None
            assert self.return_kv_values
        
        if self.use_additive_kv_cache_update:
            assert self.decode_kv_cache_as_io
            assert cache_pos_mask is not None

        bsz, seqlen, _ = x.shape

        # QKV
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        # We need view_copy elimination
        q = q.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        q, k = self.rope.forward(q, k, freqs_cos, freqs_sin)

        if self.use_kv_cache and not self.decode_kv_cache_as_io:
            assert input_pos is not None
            assert not self.return_kv_values
            output = self.SDPA(input_pos, q, k, v, bsz, seqlen, self.mask)
            return self.wo(output)

        q = q.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.return_kv_values:
            k_ret = k
            v_ret = v
        
        assert hasattr(self, "mask")
        if self.decode_kv_cache_as_io:
            assert self.use_kv_cache
            mask = self.mask[None, None, input_pos]
            if self.use_additive_kv_cache_update:
                assert cache_pos_mask is not None
                assert seqlen == 1
                k_update = cache_pos_mask * k
                v_update = cache_pos_mask * v
                k = k_cache + k_update
                v = v_cache + v_update
                assert k.shape == k_cache.shape
                assert v.shape == v_cache.shape


                # # Attempt 1 to use torch.cat:
                # # This fails to lower to ET during to_executorch due to a dynamo error related to the
                # # delegate call.  We can talk to compiler about this, but the bigger issue is although
                # # the CoreML mlpackage lowers, it fails at runtime on CPU/ANE with "input data broken / unsupported
                # # model (model code -7)".  It does run on GPU.
                # # I suspect it is related to the data-dependent / dynamic shape of k, v, and mask

                # buffer = 2 # needed to make dynamo happy
                # input_pos_item = input_pos[0].item()
                # torch._check_is_size(input_pos_item)
                # torch._check(input_pos_item + seqlen <= self.max_seq_len - buffer)
                # mask = torch.narrow(mask, dim=3, start=0, length=input_pos_item + seqlen)
                
                # k = torch.cat([torch.narrow(k_cache, dim=2, start=0, length=input_pos_item), k], axis=2)
                # v = torch.cat([torch.narrow(v_cache, dim=2, start=0, length=input_pos_item), v], axis=2)


                # # Attempt 2 to use torch.cat
                # # Dynamo fails with "expand: attempting to expand a dimension of length u0 + 1024!"
                # # I'm not confident this variant will work in CoreML if we can export it, though.
                # buffer = 2
                # input_pos_item = input_pos[0].item()
                # torch._check_is_size(input_pos_item)
                # torch._check(input_pos_item + seqlen <= self.max_seq_len - buffer)
                
                # k = torch.cat([torch.narrow(k_cache, dim=2, start=0, length=input_pos_item), k], axis=2)
                # k = k.expand(k_cache.size())
                # v = torch.cat([torch.narrow(v_cache, dim=2, start=0, length=input_pos_item), v], axis=2)
                # v = v.expand(v_cache.size())
            else:
                k = torch.ops.aten.index_put(k_cache, [None, None, input_pos, None], k)
                v = torch.ops.aten.index_put(v_cache, [None, None, input_pos, None], v)
        else:
            assert not self.use_kv_cache
            mask = self.mask[:seqlen, :seqlen]


        # grouped multiquery attention: expand out keys and values
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        output = torch.ops.coreml.sdpa(q, k, v, mask)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        output = self.wo(output)

        if self.return_kv_values:
            return output, k_ret, v_ret
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
    def __init__(self, layer_id: int, args: ModelArgs, rope: Rope):
        super().__init__()
        self.use_kv_cache = args.use_kv_cache
        self.decode_kv_cache_as_io = args.decode_kv_cache_as_io
        self.return_kv_values = (args.prefill_return_kv or args.decode_kv_cache_as_io)
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

    def forward(self, x, freqs_cos, freqs_sin, input_pos=None, k_cache=None, v_cache=None, cache_pos_mask=None):  # x: 1xN
        if self.decode_kv_cache_as_io:
            assert self.use_kv_cache
            assert k_cache is not None
            assert v_cache is not None

        if not self.return_kv_values:
            h = self.attention.forward(
                self.attention_norm(x), freqs_cos, freqs_sin, input_pos, k_cache, v_cache, cache_pos_mask,
            )
        else:
            h, k, v = self.attention.forward(
                self.attention_norm(x), freqs_cos, freqs_sin, input_pos, k_cache, v_cache, cache_pos_mask,
            )

        h = x + h
        if hasattr(self, "block_sparse_moe"):
            out = h + self.block_sparse_moe(self.ffn_norm(h))
        else:
            out = h + self.feed_forward(self.ffn_norm(h))
        
        if self.return_kv_values:
            return out, k, v
        return out


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
        self.use_kv_cache = params.use_kv_cache
        self.decode_kv_cache_as_io = params.decode_kv_cache_as_io
        self.generate_full_logits = params.generate_full_logits
        self.max_seq_len = params.max_seq_len
        self.input_prune_map = params.input_prune_map
        self.output_prune_map = params.output_prune_map

        # Whether model returns newly computed KV values
        self.return_kv_values = (params.prefill_return_kv or params.decode_kv_cache_as_io)

    def forward(
        self,
        tokens: Optional[torch.LongTensor] = None,  # tokens
        input_pos: Optional[
            torch.LongTensor
        ] = None,  # Scalar tensor indicating size of window of the caches
        k_cache: Optional[torch.FloatTensor] = None,
        v_cache: Optional[torch.FloatTensor] = None,
        cache_pos_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        h = self.tok_embeddings(tokens)
        if self.decode_kv_cache_as_io:
            assert self.use_kv_cache
            assert k_cache is not None
            assert v_cache is not None
            

            
        seqlen = h.shape[1]
        freqs_cos, freqs_sin = self.rope.get_freqs(input_pos, seqlen)

        if not self.return_kv_values:
            for layer in self.layers:
                h = layer(
                    h,
                    freqs_cos,
                    freqs_sin,
                    input_pos,
                    k_cache,
                    v_cache,
                    cache_pos_mask,
                )
        else:
            k_caches = []
            v_caches = []
            for i, layer in enumerate(self.layers):
                if not self.decode_kv_cache_as_io:
                    h, k, v = layer(
                        h,
                        freqs_cos,
                        freqs_sin,
                        input_pos,
                        k_cache,
                        v_cache,
                        cache_pos_mask,
                    )
                else:
                    h, k, v = layer(
                        h,
                        freqs_cos,
                        freqs_sin,
                        input_pos,
                        k_cache[i,:,:,:,:],
                        v_cache[i,:,:,:,:],
                        cache_pos_mask,
                    )
                k_caches.append(k)
                v_caches.append(v)
            k_ret = torch.stack(k_caches, dim=0)
            v_ret = torch.stack(v_caches, dim=0)

        if not self.generate_full_logits:
            # Only the last logit is used for the new generated token
            h = h[:, -1, :]

        h = self.norm(h)

        logits = self.output(h)

        if self.output_prune_map is not None:
            # expand to original size so that downstream applications can use the logits as-is.
            if self.generate_full_logits:
                # (1, seq_len, pruned_size) -> (1, seq_len, original_size)
                expanded_logits = torch.full(
                    [logits.shape[0], logits.shape[1], self.vocab_size],
                    float("-inf"),
                    device=logits.device,
                    dtype=logits.dtype,
                )
                expanded_logits[:, :, list(self.output_prune_map.values())] = logits
            else:
                # (1, pruned_size) -> (1, original_size)
                expanded_logits = torch.full(
                    [logits.shape[0], self.vocab_size],
                    float("-inf"),
                    device=logits.device,
                    dtype=logits.dtype,
                )
                expanded_logits[:, list(self.output_prune_map.values())] = logits
            logits = expanded_logits

        if self.return_kv_values:
            return logits, k_ret, v_ret
        return logits
