from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Type, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from executorch.examples.models.llama.lora import LoRALinear
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.norm import RMSNorm
from executorch.examples.models.llama.rope import Rope


class ForwardOptions(TypedDict, total=False):
    """Optional parameters for `Attention.forward` (compative with Python 3.10 and plus)."""

    mask: Optional[torch.Tensor]
    input_pos: Optional[torch.Tensor]
    freqs_cos_override: Optional[torch.Tensor]
    freqs_sin_override: Optional[torch.Tensor]
    in_cache_state: Optional[Any]
    out_cache_state: Optional[Any]
    last_valid_token_pos: Optional[torch.LongTensor]


class Attention(nn.Module, ABC):
    """Abstract base class for attention mechanisms with unified interface."""

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        **kwargs: ForwardOptions,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """Forward pass for attention mechanism.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            freqs_cos, freqs_sin: Rotary position embedding frequencies
            ForwardOptions: grouped optional args

        Returns:
            Tuple of (output tensor, updated cache state)
        """
        pass


ATTENTION_REGISTRY: Dict[str, Type[Attention]] = {}


def register_attention(name: str):
    """Decorator to register attention classes"""

    def decorator(cls: Type[Attention]):
        ATTENTION_REGISTRY[name.lower()] = cls
        return cls

    return decorator


class KVCache(nn.Module):
    def __init__(
        self,
        max_batch_size: int,
        max_context_length: int,
        n_heads: int,
        head_dim: int,
        enable_dynamic_shape: bool,
        dtype=torch.float32,
    ):
        super().__init__()
        self.max_context_length = max_context_length
        cache_shape = (max_batch_size, n_heads, max_context_length, head_dim)

        self.max_batch_size = max_batch_size
        self.n_heads = n_heads
        self.head_dim = head_dim
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
        # input_pos: [S], k_val: [B, H, S, D]
        if self.enable_dynamic_shape:
            start_pos = input_pos[0].item()
            torch._check_is_size(start_pos)
            torch._check(start_pos < self.max_context_length)
            dim_to_slice = 2
            seq_length = k_val.size(dim_to_slice)
            indices = torch.arange(seq_length) + start_pos
            self.k_cache.index_copy_(dim_to_slice, indices, k_val)
            self.v_cache.index_copy_(dim_to_slice, indices, v_val)
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
        dim: int,
        head_dim: int,
        n_rep: int,
        max_context_len: int,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.n_rep = n_rep
        self.max_context_len = max_context_len

    def forward(
        self,
        input_pos: torch.Tensor,
        q: torch.Tensor,  # Already have rotary embeddings. (bs, n_local_heads, seqlen, head_dim)
        k: torch.Tensor,  # Already have rotary embeddings. (bs, n_local_kv_heads, seqlen, head_dim)
        v: torch.Tensor,  # (bs, n_local_kv_heads, seqlen, head_dim)
        bsz,
        seqlen,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        # TODO(kimishpatel): This should not be necessary because scaled_dot_product_attention
        # can natively support GQA now. But needs enable_gqa=True
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        return y.transpose(1, 2).reshape(bsz, seqlen, self.dim)


def _create_causal_mask_for_ring_buffer(
    cache_positions, window_size, start_pos, seq_len
):
    pos_q = start_pos + torch.arange(seq_len, dtype=torch.long).view(-1, 1)
    delta = pos_q - cache_positions
    attn_mask = (cache_positions >= 0) & (delta >= 0) & (delta < window_size)
    attn_mask = torch.where(attn_mask == True, 0, float("-inf"))  # noqa E712
    return attn_mask


class CacheUpdateStrategy(Enum):
    RING_BUFFER = "RingBuffer"
    INVALID = "Invalid"


class CachePositionsManager(nn.Module):
    def __init__(
        self,
        max_context_length: int,
        cache_update_strategy: CacheUpdateStrategy = CacheUpdateStrategy.RING_BUFFER,
    ):
        super().__init__()
        assert (
            cache_update_strategy == CacheUpdateStrategy.RING_BUFFER
        ), "Only RingBuffer is supported"
        self.max_context_length = max_context_length
        self.register_buffer(
            "cache_positions",
            torch.zeros((self.max_context_length), dtype=torch.long, device="cpu"),
        )

    def calculate_positions_and_update_indices(self, input_pos: torch.Tensor, seq_len):
        """
        Calculate indices, into k_cache, v_cache, where to put k_val tensor.
        Given the input_pos and length of k_val at sequence dim, the input pos may
        have to wrap around if it is smaller than the cache capacity.
        If it is larger than the cache capacity then just pick the last
        self.max_context_length entries.

        Additionally:
        Update the cache positions buffer with the new indices.
        Given the cache positions in sequence dim, indicated by indices,
        we can just update cache_positions buffer using orig_indices.
        For example
        Given cache capacity of 4 and update of length 3 with start_pos = 2
        will have following values
        indices = [2, 3, 0]
        orig_indices = [2, 3, 4]
        So cache_positions after the update will be [4, 1, 2, 3]
        Note cache_positions[1] = 1 that is from previous write to the cache.
        The corner case here is cache positions before cache rolls over.
        For example when start_pos = 0 and update is of length 2, then we have
        filled positions 0 and 1 in the buffer, while the rest are invalid. In this case
        we have
        indices = [0, 1]
        orig_indices = [0, 1]
        But if we have cache_positins = [0, 1, 0, 0] that is not valid. Hence we have
        to make sure that invalid positions have a sentinel value of - 1.
        """
        start_pos = input_pos[0].item()
        torch._check_is_size(start_pos)
        orig_indices = torch.arange(seq_len, dtype=torch.long) + start_pos
        indices = orig_indices % self.max_context_length

        full_t = torch.full((self.max_context_length,), -1, dtype=torch.long)
        arange_tensor = torch.arange(self.max_context_length, dtype=torch.long)
        cache_positions = torch.where(
            arange_tensor < start_pos, self.cache_positions, full_t
        )
        self.cache_positions.copy_(cache_positions)
        self.cache_positions.index_copy_(0, indices, orig_indices)

        return indices


class RingKVCache(KVCache):
    def __init__(
        self,
        max_batch_size: int,
        max_context_length: int,
        n_heads: int,
        head_dim: int,
        enable_dynamic_shape: bool,
        dtype=torch.float32,
    ):
        self.window_size = max_context_length
        """
        Reason why we want the kv cache size to be twice the context length:
        Sliding window attention without ringbuffer
        pos   0  1  2  3  4  5  6  7  8  9  10
        0     x  0  0  0  0  0  0  0  0  0  0
        1     x  x  0  0  0  0  0  0  0  0  0
        2     x  x  x  0  0  0  0  0  0  0  0
        3     x  x  x  x  0  0  0  0  0  0  0
        4     0  x  x  x  x  0  0  0  0  0  0
        5     0  0  x  x  x  x  0  0  0  0  0
        6     0  0  0  x  x  x  x  0  0  0  0
        7     0  0  0  0  x  x  x  x  0  0  0
        8     0  0  0  0  0  x  x  x  x  0  0
        9     0  0  0  0  0  0  x  x  x  x  0
        10    0  0  0  0  0  0  0  x  x  x  x

        So when doing attention for pos = 5 and seq_len = 4 our attention
        mask would be
        5     0  0  x  x  x  x  0  0  0  0  0
        6     0  0  0  x  x  x  x  0  0  0  0
        7     0  0  0  0  x  x  x  x  0  0  0
        8     0  0  0  0  0  x  x  x  x  0  0
        Thus tok at pos = 5 is able to attend to tokens at pos 2, 3 and 4.
        This is how training is done.

        Now lets consider ring kv cache of size 4. When we are at pos = 5
        before updating the kv cache, state of the kv cache would be
        [4 1 2 3]. That is we evicted token at pos = 0 out. Now during
        attention calculation at pos = 5 seq len = 4, we will update cache and
        new pos in the cache would be [8 5 6 7]. So note that 5 can now only attend
        to itself. Not 2, 3 and 4 as you would have during training.
        So not having kept 2, 3 and 4 in cache means we will have divergent behavior.
        Worst case of this would have been when update it equal to the length of
        the cache. like in our case pos = 5 seq len = 4.
        Thus we need to have a cache that is larger. How much larger, as much as
        the sliding window size. So twice the max_context_length.
        How would that have helped. Lets see. At pos = 5 our cache would have
        [0, 1, 2, 3, 4, NA, NA, NA] After cache update we would have
        [8, 1, 2, 3, 4, 5, 6, 7]. We kicked out token at pos = 0. However, the
        current step still has access to [pos - sliding_window_size, pos] tokens.
        
        To make sure we dont over attend, i.e. we dont have pos = 5
        to attend to pos = 1, mask calculaton has to account for the sliding window
        size.
        """
        super().__init__(
            max_batch_size,
            max_context_length * 2,
            n_heads,
            head_dim,
            enable_dynamic_shape,
            dtype,
        )
        self.cache_positions_manager = CachePositionsManager(self.max_context_length)
        self.is_ring_buffer = True

    def create_causal_mask_for_ring_buffer(self, start_pos, seq_len):
        cache_positions = self.cache_positions_manager.cache_positions
        return _create_causal_mask_for_ring_buffer(
            cache_positions, self.window_size, start_pos, seq_len
        )

    def update(
        self, input_pos: torch.Tensor, k_val: torch.Tensor, v_val: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # input_pos: [S], k_val: [B, H, S, D]
        seq_len = k_val.size(2)
        assert seq_len <= self.k_cache.size(
            2
        ), f"Update sequence length({seq_len}) for kv cache must be smaller than the cache size({self.k_cache.size(2)})"
        indices = self.cache_positions_manager.calculate_positions_and_update_indices(
            input_pos, seq_len
        )
        if self.enable_dynamic_shape:
            start_pos = input_pos[0].item()
            torch._check_is_size(start_pos)

            self.k_cache.index_copy_(2, indices, k_val)
            self.v_cache.index_copy_(2, indices, v_val)
        else:
            self.k_cache[:, :, indices] = k_val
            self.v_cache[:, :, indices] = v_val

        return self.k_cache, self.v_cache


@register_attention("mha")
class AttentionMHA(Attention):
    def __init__(
        self,
        args: ModelArgs,
        layer_id: int,
        rope: Rope,
        **_kwargs: Any,
    ):
        """
        Multi-head attention layer.

        Args:
            args (ModelArgs): Model configuration parameters.
            layer_id (int): Layer index.
            rope (Rope): Rotary position embedding module.
        """
        super().__init__()
        self.use_kv_cache = args.use_kv_cache
        self.n_heads = args.n_heads
        self.n_kv_heads = self.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert self.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = self.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.head_dim
        self.max_batch_size = args.max_batch_size
        self.max_context_len = args.max_context_len
        self.dim = args.dim
        self.attention_qkv_bias = args.attention_qkv_bias
        self.use_qk_norm = args.use_qk_norm
        self.qk_norm_before_rope = args.qk_norm_before_rope
        self.enable_dynamic_shape = args.enable_dynamic_shape

        if self.use_qk_norm:
            q_norm_dim = self.head_dim
            k_norm_dim = self.head_dim
            self.q_norm_fn = RMSNorm(
                q_norm_dim,
                eps=args.norm_eps,
                add_unit_offset=args.rms_norm_add_unit_offset,
            )
            self.k_norm_fn = RMSNorm(
                k_norm_dim,
                eps=args.norm_eps,
                add_unit_offset=args.rms_norm_add_unit_offset,
            )

        self.wq = (
            LoRALinear(
                in_dim=args.dim,
                out_dim=args.n_heads * args.head_dim,
                rank=args.r,
                alpha=args.lora_alpha,
                dropout=0.0,
                use_bias=args.attention_qkv_bias,
            )
            if args.target_modules is not None and "q_proj" in args.target_modules
            else nn.Linear(
                self.dim, self.n_heads * self.head_dim, bias=self.attention_qkv_bias
            )
        )
        self.wk = (
            LoRALinear(
                in_dim=args.dim,
                out_dim=args.n_kv_heads * args.head_dim,
                rank=args.r,
                alpha=args.lora_alpha,
                dropout=0.0,
                use_bias=args.attention_qkv_bias,
            )
            if args.target_modules is not None and "k_proj" in args.target_modules
            else nn.Linear(
                self.dim, self.n_kv_heads * self.head_dim, bias=self.attention_qkv_bias
            )
        )
        self.wv = (
            LoRALinear(
                in_dim=args.dim,
                out_dim=args.n_kv_heads * args.head_dim,
                rank=args.r,
                alpha=args.lora_alpha,
                dropout=0.0,
                use_bias=args.attention_qkv_bias,
            )
            if args.target_modules is not None and "v_proj" in args.target_modules
            else nn.Linear(
                self.dim, self.n_kv_heads * self.head_dim, bias=self.attention_qkv_bias
            )
        )
        self.wo = (
            LoRALinear(
                in_dim=args.n_heads * args.head_dim,
                out_dim=args.dim,
                rank=args.r,
                alpha=args.lora_alpha,
                dropout=0.0,
                use_bias=args.attention_qkv_bias,
            )
            if args.target_modules is not None
            and (
                "output_proj" in args.target_modules or "o_proj" in args.target_modules
            )
            else nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)
        )

        self.layer_id = layer_id

        self.rope = rope

        causal_mask = torch.tril(
            torch.ones(
                self.max_context_len,
                self.max_context_len,
                dtype=torch.bool,
                device="cpu",
            )
        )
        self.register_buffer("mask", causal_mask, persistent=False)

        if self.use_kv_cache:
            self.kv_cache = KVCache(
                args.max_batch_size,
                args.max_context_len,
                self.n_kv_heads,
                self.head_dim,
                args.enable_dynamic_shape,
            )
            self.SDPA = SDPA(
                dim=self.n_local_heads * self.head_dim,
                head_dim=self.head_dim,
                n_rep=self.n_rep,
                max_context_len=self.max_context_len,
            )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        **kwargs: ForwardOptions,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        input_pos = kwargs.get("input_pos")
        bsz, seqlen, _ = x.shape

        # QKV
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        # We need view_copy elimination
        q = q.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        if self.use_qk_norm and self.qk_norm_before_rope:
            q = self.q_norm_fn(q)
            k = self.k_norm_fn(k)

        # RoPE relative positional embeddings
        q, k = self.rope.forward(q, k, freqs_cos, freqs_sin)

        q = q.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_qk_norm and not self.qk_norm_before_rope:
            q = self.q_norm_fn(q)
            k = self.k_norm_fn(k)

        if self.use_kv_cache:
            assert input_pos is not None
            if self.enable_dynamic_shape:
                start_pos = input_pos[-1].item()
                torch._check_is_size(start_pos)
                torch._check(start_pos < self.max_context_len)
                seq_length = q.size(2)
                # pyre-ignore: Incompatible parameter type [6]
                attn_mask = self.mask.narrow(0, start_pos, seq_length)
            else:
                # mask is always 2D
                attn_mask = self.mask[input_pos]
            k, v = self.kv_cache.update(input_pos, k, v)
            if getattr(self.kv_cache, "is_ring_buffer", False):
                attn_mask = self.kv_cache.create_causal_mask_for_ring_buffer(
                    input_pos[0].item(), seqlen
                )
            output = self.SDPA(input_pos, q, k, v, bsz, seqlen, attn_mask)
            return self.wo(output), None

        # grouped multiquery attention: expand out keys and values
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        assert hasattr(self, "mask")

        mask = self.mask[:seqlen, :seqlen]

        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        output = output.transpose(1, 2).reshape(bsz, seqlen, -1)

        output = self.wo(output)

        return output, None


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


class RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


@register_attention("qwen3_5_full")
class AttentionQwen3_5Full(Attention):
    """Qwen3.5 full-attention block with q-gating."""

    def __init__(
        self,
        args: ModelArgs,
        layer_id: int,
        rope: Rope,
        **_kwargs: Any,
    ):
        super().__init__()
        self.use_kv_cache = args.use_kv_cache
        self.n_heads = args.n_heads
        self.n_kv_heads = self.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert self.n_heads % self.n_kv_heads == 0
        self.n_local_heads = self.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.head_dim
        self.max_context_len = args.max_context_len
        self.dim = args.dim
        self.attention_qkv_bias = args.attention_qkv_bias
        self.use_qk_norm = args.use_qk_norm
        self.qk_norm_before_rope = args.qk_norm_before_rope
        self.enable_dynamic_shape = args.enable_dynamic_shape

        if self.use_qk_norm:
            self.q_norm_fn = RMSNorm(
                self.head_dim,
                eps=args.norm_eps,
                add_unit_offset=args.rms_norm_add_unit_offset,
            )
            self.k_norm_fn = RMSNorm(
                self.head_dim,
                eps=args.norm_eps,
                add_unit_offset=args.rms_norm_add_unit_offset,
            )

        self.wq = nn.Linear(
            self.dim, self.n_heads * self.head_dim * 2, bias=self.attention_qkv_bias
        )
        self.wk = nn.Linear(
            self.dim, self.n_kv_heads * self.head_dim, bias=self.attention_qkv_bias
        )
        self.wv = nn.Linear(
            self.dim, self.n_kv_heads * self.head_dim, bias=self.attention_qkv_bias
        )
        self.wo = nn.Linear(
            self.n_heads * self.head_dim,
            self.dim,
            bias=False,
        )

        self.layer_id = layer_id
        self.rope = rope

        causal_mask = torch.tril(
            torch.ones(
                self.max_context_len,
                self.max_context_len,
                dtype=torch.bool,
                device="cpu",
            )
        )
        self.register_buffer("mask", causal_mask, persistent=False)

        if self.use_kv_cache:
            self.kv_cache = KVCache(
                args.max_batch_size,
                args.max_context_len,
                self.n_kv_heads,
                self.head_dim,
                args.enable_dynamic_shape,
            )
            self.SDPA = SDPA(
                dim=self.n_local_heads * self.head_dim,
                head_dim=self.head_dim,
                n_rep=self.n_rep,
                max_context_len=self.max_context_len,
            )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        **kwargs: ForwardOptions,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        input_pos = kwargs.get("input_pos")
        bsz, seqlen, _ = x.shape

        # Q and gate are packed in q_proj output.
        q_and_gate = self.wq(x).view(bsz, seqlen, self.n_local_heads, self.head_dim * 2)
        q, gate = torch.chunk(q_and_gate, 2, dim=-1)
        gate = gate.reshape(bsz, seqlen, -1)

        k = self.wk(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        if self.use_qk_norm and self.qk_norm_before_rope:
            q = self.q_norm_fn(q)
            k = self.k_norm_fn(k)

        q, k = self.rope.forward(q, k, freqs_cos, freqs_sin)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_qk_norm and not self.qk_norm_before_rope:
            q = self.q_norm_fn(q)
            k = self.k_norm_fn(k)

        if self.use_kv_cache:
            assert input_pos is not None
            if self.enable_dynamic_shape:
                start_pos = input_pos[-1].item()
                torch._check_is_size(start_pos)
                torch._check(start_pos < self.max_context_len)
                seq_length = q.size(2)
                attn_mask = self.mask.narrow(0, start_pos, seq_length)
            else:
                attn_mask = self.mask[input_pos]
            k, v = self.kv_cache.update(input_pos, k, v)
            if getattr(self.kv_cache, "is_ring_buffer", False):
                attn_mask = self.kv_cache.create_causal_mask_for_ring_buffer(
                    input_pos[0].item(), seqlen
                )
            output = self.SDPA(input_pos, q, k, v, bsz, seqlen, attn_mask)
            output = output * torch.sigmoid(gate)
            return self.wo(output), None

        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        mask = self.mask[:seqlen, :seqlen]
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        output = output.transpose(1, 2).reshape(bsz, seqlen, -1)
        output = output * torch.sigmoid(gate)
        return self.wo(output), None


@register_attention("gated_deltanet")
class AttentionGatedDeltaNet(Attention):
    """Qwen3.5 linear-attention (Gated DeltaNet) block with internal state."""

    def __init__(
        self,
        args: ModelArgs,
        layer_id: int,
        rope: Rope,
        **_kwargs: Any,
    ):
        super().__init__()
        del rope  # DeltaNet layers do not use RoPE.

        self.hidden_size = args.dim
        self.max_batch_size = args.max_batch_size
        self.layer_id = layer_id

        assert args.linear_num_key_heads is not None
        assert args.linear_num_value_heads is not None
        assert args.linear_key_head_dim is not None
        assert args.linear_value_head_dim is not None

        self.num_k_heads = args.linear_num_key_heads
        self.num_v_heads = args.linear_num_value_heads
        self.head_k_dim = args.linear_key_head_dim
        self.head_v_dim = args.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = args.linear_conv_kernel_dim

        assert (
            self.num_v_heads % self.num_k_heads == 0
        ), "linear_num_value_heads must be divisible by linear_num_key_heads."
        self.head_repeat = self.num_v_heads // self.num_k_heads

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
            padding=0,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.norm = RMSNormGated(self.head_v_dim, eps=args.norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.register_buffer(
            "conv_state",
            torch.zeros(
                self.max_batch_size,
                self.conv_dim,
                self.conv_kernel_size,
                dtype=torch.float32,
                device="cpu",
            ),
        )
        self.register_buffer(
            "recurrent_state",
            torch.zeros(
                self.max_batch_size,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                dtype=torch.float32,
                device="cpu",
            ),
        )

    def _maybe_reset_state(
        self, input_pos: Optional[torch.Tensor], batch_size: int
    ) -> None:
        if input_pos is None:
            return
        reset = (input_pos[0] == 0).to(self.conv_state.dtype)
        keep = 1.0 - reset
        self.conv_state[:batch_size].mul_(keep)
        self.recurrent_state[:batch_size].mul_(keep)

    def _apply_causal_conv(self, mixed_qkv: torch.Tensor) -> torch.Tensor:
        # mixed_qkv: (batch, seq_len, conv_dim)
        batch_size, seq_len, _ = mixed_qkv.shape
        mixed_qkv = mixed_qkv.transpose(1, 2)
        state_len = self.conv_state.shape[-1]
        hidden_states_new = torch.cat([self.conv_state[:batch_size], mixed_qkv], dim=-1)
        new_conv_state = hidden_states_new[:, :, -state_len:]
        with torch.no_grad():
            self.conv_state[:batch_size].copy_(new_conv_state.to(self.conv_state.dtype))
        out = F.conv1d(
            hidden_states_new,
            self.conv1d.weight,
            self.conv1d.bias,
            padding=0,
            groups=self.conv_dim,
        )
        out = F.silu(out[:, :, -seq_len:]).to(mixed_qkv.dtype)
        return out.transpose(1, 2).contiguous()

    def _recurrent_gated_delta_rule(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        # query/key/value: (batch, seq_len, num_heads, head_dim)
        # g/beta: (batch, seq_len, num_heads)
        initial_dtype = query.dtype
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(torch.float32)
            for x in (query, key, value, beta, g)
        ]

        batch_size, num_heads, sequence_length, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        scale = 1.0 / (query.shape[-1] ** 0.5)
        query = query * scale

        core_attn_out = torch.zeros(
            batch_size,
            num_heads,
            sequence_length,
            v_head_dim,
            device=value.device,
            dtype=value.dtype,
        )
        last_recurrent_state = self.recurrent_state[:batch_size].to(value.dtype)

        for i in range(sequence_length):
            q_t = query[:, :, i]
            k_t = key[:, :, i]
            v_t = value[:, :, i]
            g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, :, i].unsqueeze(-1)

            last_recurrent_state = last_recurrent_state * g_t
            kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            last_recurrent_state = last_recurrent_state + k_t.unsqueeze(
                -1
            ) * delta.unsqueeze(-2)
            core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(
                dim=-2
            )

        with torch.no_grad():
            self.recurrent_state[:batch_size].copy_(
                last_recurrent_state.to(self.recurrent_state.dtype)
            )

        return core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        **kwargs: ForwardOptions,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        del freqs_cos
        del freqs_sin
        input_pos = kwargs.get("input_pos")
        batch_size, seq_len, _ = x.shape
        assert (
            batch_size <= self.max_batch_size
        ), f"batch_size ({batch_size}) exceeds max_batch_size ({self.max_batch_size})"

        self._maybe_reset_state(input_pos, batch_size)

        mixed_qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)

        mixed_qkv = self._apply_causal_conv(mixed_qkv)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        if self.head_repeat > 1:
            query = query.repeat_interleave(self.head_repeat, dim=2)
            key = key.repeat_interleave(self.head_repeat, dim=2)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        core_attn_out = self._recurrent_gated_delta_rule(query, key, value, g, beta)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        return self.out_proj(core_attn_out), None


@register_attention("skip")
class AttentionSkip(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        **kwargs: ForwardOptions,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        return x, None
