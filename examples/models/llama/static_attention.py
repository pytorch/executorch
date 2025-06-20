from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from executorch.examples.models.llama.attention import (
    Attention,
    AttentionMHA,
    ForwardOptions,
    register_attention,
)
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.rope import Rope


_CacheMap = Dict[str, torch.Tensor]
# Key and value caches are kept separate so the key caches can be kept transposed.
_InputCacheState = Tuple[_CacheMap, _CacheMap]
_OutputCacheState = Tuple[_CacheMap, _CacheMap]


class StaticKVCache(nn.Module, ABC):
    def __init__(self, layer_id: int, head_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.head_id = head_id

    @abstractmethod
    def update(
        self,
        new_data: torch.Tensor,
        in_cache_state: Optional[_InputCacheState],
        out_cache_state: Optional[_OutputCacheState],
    ) -> Tuple[torch.Tensor, Optional[_OutputCacheState]]:
        """
        Given input cache state and new keys/values, returns the combined keys/values
        and the updated the output cache state.
        """
        pass

    def cache_key(self) -> str:
        return self.calculate_cache_key(self.layer_id, self.head_id)

    @staticmethod
    def calculate_cache_key(layer_id: int, head_id: int) -> str:
        return f"l{layer_id},h{head_id}"

    @staticmethod
    def apply_update(
        cache, update, pos, style, transpose=False, update_pos=0, update_len=None
    ):
        """
        After inference, update the cache state for next iteration. The runtime needs to
        implement the same operation.
        """
        if style == "shift_pointer":
            if transpose:
                update_len = update_len or update.size(-1)
                updated = torch.roll(cache, -update_len, -1)
                updated[:, :, -update_len:] = update[
                    :, :, update_pos : update_pos + update_len
                ]
            else:
                update_len = update_len or update.size(-2)
                updated = torch.roll(cache, -update_len, -2)
                updated[:, -update_len:, :] = update[
                    :, update_pos : update_pos + update_len, :
                ]

        if style == "smart_mask":
            updated = torch.clone(cache)
            if transpose:
                update_len = update_len or update.size(-1)
                updated[:, :, pos : pos + update_len] = update[
                    :, :, update_pos : update_pos + update_len
                ]
            else:
                update_len = update_len or update.size(-2)
                updated[:, pos : pos + update_len, :] = update[
                    :, update_pos : update_pos + update_len, :
                ]

        return updated


class StaticKCache(StaticKVCache):
    def __init__(self, layer_id: int, head_id: int, transpose=False):
        """
        If transpose is True, key cache is kept in (batch, dim, seq_len), otherwise in
        (batch, seq_len, dim).
        """
        super().__init__(layer_id, head_id)
        self.transpose = transpose

    def update(
        self,
        new_data: torch.Tensor,
        in_cache_state: Optional[_InputCacheState],
        out_cache_state: Optional[_OutputCacheState],
    ) -> Tuple[torch.Tensor, Optional[_OutputCacheState]]:
        seq_dim = -2
        if self.transpose:
            seq_dim = -1
            new_data = new_data.transpose(-1, -2)
        if in_cache_state is None:
            return new_data, None
        if out_cache_state is None:
            out_cache_state = ({}, {})

        all_data = torch.cat(
            [in_cache_state[0][self.cache_key()], new_data], dim=seq_dim
        )
        out_k_cache, out_v_cache = out_cache_state
        out_k_cache[self.cache_key()] = new_data
        return all_data, (out_k_cache, out_v_cache)


class StaticVCache(StaticKVCache):
    def update(
        self,
        new_data: torch.Tensor,
        in_cache_state: Optional[_InputCacheState],
        out_cache_state: Optional[_OutputCacheState],
    ) -> Tuple[torch.Tensor, Optional[_OutputCacheState]]:
        if in_cache_state is None:
            return new_data, None
        if out_cache_state is None:
            out_cache_state = ({}, {})

        all_data = torch.cat([in_cache_state[1][self.cache_key()], new_data], dim=-2)
        out_k_cache, out_v_cache = out_cache_state
        out_v_cache[self.cache_key()] = new_data
        return all_data, (out_k_cache, out_v_cache)


class StaticAttentionMask:
    def __init__(self, input_len, cache_len, style, mask_val=float("-inf")):
        self.input_len = input_len
        self.cache_len = cache_len
        assert style in ("shift_pointer", "smart_mask")
        self.style = style
        self.mask_val = mask_val
        self.unmasked_len = 0
        self.tensor = torch.zeros(1, input_len, input_len + cache_len)
        self.reset()

    def reset(self):
        self.unmasked_len = 0
        self.tensor[:, :, : self.cache_len] = self.mask_val

    def unmask(self, new_unmasked_len):
        if new_unmasked_len <= 0:
            return

        if self.style == "shift_pointer":
            self.tensor[
                :,
                :,
                self.cache_len
                - self.unmasked_len
                - new_unmasked_len : self.cache_len
                - self.unmasked_len,
            ] = 0

        if self.style == "smart_mask":
            self.tensor[
                :,
                :,
                self.unmasked_len : self.unmasked_len + new_unmasked_len,
            ] = 0

        self.unmasked_len += new_unmasked_len


class StaticAttentionIOManager:
    def __init__(
        self,
        config: ModelArgs,
        input_len: int,
        cache_len: int,
        style: str = "shift_pointer",
        mask_val: float = float("-inf"),
    ):
        self.mask = StaticAttentionMask(
            input_len, cache_len, style=style, mask_val=mask_val
        )

        rope = Rope(config)
        freqs = rope.get_freqs(None, config.max_seq_len)
        self.freqs_cos = freqs[0]
        self.freqs_sin = freqs[1]

        self.k_caches = {
            StaticKVCache.calculate_cache_key(layer_id, head_id): torch.zeros(
                1, cache_len, config.head_dim
            )
            for layer_id in range(config.n_layers)
            for head_id in range(config.n_kv_heads)
        }
        self.v_caches = {
            StaticKVCache.calculate_cache_key(layer_id, head_id): torch.zeros(
                1, cache_len, config.head_dim
            )
            for layer_id in range(config.n_layers)
            for head_id in range(config.n_kv_heads)
        }

        self.config = config
        self.input_len = input_len
        self.cache_len = cache_len
        self.style = style
        self.mask_val = mask_val
        self.pos = 0
        self.cache_full = False

    def reset(self):
        self.pos = 0
        self.cache_full = False
        self.mask.reset()

    def prefill(
        self,
        model: Callable[..., Any],
        tokens: List[int],
    ) -> torch.Tensor:
        if self.cache_full:
            raise RuntimeError("KV cache is full.")

        self.mask.tensor[:, :, self.cache_len :] = torch.triu(
            torch.full((1, self.input_len, self.input_len), self.mask_val),
            diagonal=1,
        )

        logits = None
        all_logits = None
        for i in range(0, len(tokens), self.input_len):
            logits = self._run_once(model, tokens[i : i + self.input_len])[0]
            if self.config.generate_full_logits:
                if all_logits is None:
                    all_logits = logits
                else:
                    all_logits = torch.cat([all_logits, logits], dim=1)

        if self.config.generate_full_logits:
            return all_logits[:, : len(tokens), :]

        return logits

    def decode(
        self,
        model: Callable[..., Any],
        init_token: int,
        n: int,
        stop_tokens: Optional[List[int]] = None,
    ):
        if self.cache_full:
            raise RuntimeError("KV cache is full.")

        self.mask.tensor[:, :, self.cache_len :] = torch.triu(
            torch.full((1, self.input_len, self.input_len), self.mask_val),
            diagonal=1,
        )

        stop_tokens = stop_tokens or []
        new_tokens = [init_token]
        for _ in range(n):
            y = self._run_once(model, new_tokens[-1:])[0]
            new_tokens.append(y[:, :1, :].argmax().item())
            if new_tokens[-1] in stop_tokens:
                break

        return new_tokens

    def _run_once(
        self,
        model: Callable[..., Any],
        tokens: List[int],
        non_padded_len: Optional[int] = None,
        freqs_cos_override: Optional[torch.Tensor] = None,
        freqs_sin_override: Optional[torch.Tensor] = None,
    ):
        n_tokens = len(tokens)
        if n_tokens < self.input_len:
            tokens += [0] * (self.input_len - n_tokens)
        tokens = torch.tensor([tokens], dtype=torch.int32)
        if freqs_cos_override is None:
            freqs_cos_override = self.freqs_cos[self.pos : self.pos + self.input_len]
        if freqs_sin_override is None:
            freqs_sin_override = self.freqs_sin[self.pos : self.pos + self.input_len]
        y, attn_updates = model(
            tokens,
            {
                "mask": self.mask.tensor,
                "freqs_cos_override": freqs_cos_override,
                "freqs_sin_override": freqs_sin_override,
                "in_cache_state": (self.k_caches, self.v_caches),
            },
        )
        non_padded_len = non_padded_len or n_tokens
        if self.pos + non_padded_len <= self.cache_len:
            self._update_states(attn_updates, 0, non_padded_len)
        else:
            self.cache_full = True

        return y, attn_updates

    def _update_states(self, attn_updates, update_pos, update_len):
        assert self.pos + update_len <= self.cache_len

        self.mask.unmask(update_len)
        k_cache_updates, v_cache_updates = attn_updates["out_cache_state"]
        for cache_id, update in k_cache_updates.items():
            self.k_caches[cache_id] = StaticKVCache.apply_update(
                self.k_caches[cache_id],
                update,
                self.pos,
                style=self.style,
                update_pos=update_pos,
                update_len=update_len,
            )
        for cache_id, update in v_cache_updates.items():
            self.v_caches[cache_id] = StaticKVCache.apply_update(
                self.v_caches[cache_id],
                update,
                self.pos,
                style=self.style,
                update_pos=update_pos,
                update_len=update_len,
            )
        self.pos += update_len


class _Rope(nn.Module):
    def __init__(self, use_hf_rope):
        super().__init__()
        self.use_hf_rope = use_hf_rope

    def forward(
        self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
    ) -> torch.Tensor:
        if self.use_hf_rope:
            if len(freqs_cos.shape) == 2:
                freqs_cos = freqs_cos.unsqueeze(0)
            if len(freqs_sin.shape) == 2:
                freqs_sin = freqs_sin.unsqueeze(0)
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            x_rotated = torch.cat((-x2, x1), dim=-1)
            return x * freqs_cos + x_rotated * freqs_sin
        else:
            x_r, x_i = x[..., ::2], x[..., 1::2]
            x_out_r = x_r * freqs_cos - x_i * freqs_sin
            x_out_i = x_r * freqs_sin + x_i * freqs_cos
            x_out = torch.cat([x_out_r, x_out_i], dim=-1)
            return x_out


@register_attention("static")
class StaticAttention(Attention):
    """
    An attention implementation meant for NPUs that require static shapes and are not
    flexible with tensor operations needed to perform KV cache updates. MHA/GQA is
    implemented as multiple SHAs, and the KV caches keep valid data at the end so the
    model only needs to perform a concat to combine past and new data.
    """

    def __init__(self, config: ModelArgs, layer_id: int, rope: Rope):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = (
            self.n_heads if config.n_kv_heads is None else config.n_kv_heads
        )
        assert self.n_heads % self.n_kv_heads == 0
        self.n_heads_per_kv_group = self.n_heads // self.n_kv_heads
        self.dim = config.dim
        self.head_dim = config.head_dim
        self.inv_scale = 1.0 / (float(self.head_dim) ** 0.5)
        self.attention_qkv_bias = config.attention_qkv_bias
        self.use_qk_norm = config.use_qk_norm
        self.use_conv2d = False

        self.wqs = nn.ModuleList(
            [
                nn.Linear(self.dim, self.head_dim, bias=self.attention_qkv_bias)
                for _ in range(self.n_heads)
            ]
        )
        self.wks = nn.ModuleList(
            [
                nn.Linear(self.dim, self.head_dim, bias=self.attention_qkv_bias)
                for _ in range(self.n_kv_heads)
            ]
        )
        self.wvs = nn.ModuleList(
            [
                nn.Linear(self.dim, self.head_dim, bias=self.attention_qkv_bias)
                for _ in range(self.n_kv_heads)
            ]
        )

        self.k_caches = nn.ModuleList(
            [StaticKCache(layer_id, i) for i in range(self.n_kv_heads)]
        )
        self.v_caches = nn.ModuleList(
            [StaticVCache(layer_id, i) for i in range(self.n_kv_heads)]
        )
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)
        self.rope = _Rope(rope.params.use_hf_rope)

        if self.use_qk_norm:
            self.q_norm = torch.nn.RMSNorm(self.head_dim, config.norm_eps)
            self.k_norm = torch.nn.RMSNorm(self.head_dim, config.norm_eps)
        else:
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        **kwargs: ForwardOptions,
    ):
        mask = kwargs.get("mask")
        if (freqs_cos_override := kwargs.get("freqs_cos_override")) is not None:
            freqs_cos = freqs_cos_override  # pyre-ignore
        if (freqs_sin_override := kwargs.get("freqs_sin_override")) is not None:
            freqs_sin = freqs_sin_override  # pyre-ignore
        in_cache_state = kwargs.get("in_cache_state")
        out_cache_state = kwargs.get("out_cache_state")

        bsz, seq_len, dim = x.shape
        if self.use_conv2d:
            x = x.reshape(bsz, seq_len, 1, dim).transpose(1, 3)

        new_qs = [self.wqs[i](x) for i in range(self.n_heads)]
        new_ks = [self.wks[i](x) for i in range(self.n_kv_heads)]
        new_vs = [self.wvs[i](x) for i in range(self.n_kv_heads)]

        if self.use_conv2d:

            def from_conv2ds(ts):
                return [
                    t.reshape(bsz, self.head_dim, seq_len).transpose(1, 2) for t in ts
                ]

            new_qs = from_conv2ds(new_qs)
            new_ks = from_conv2ds(new_ks)
            new_vs = from_conv2ds(new_vs)

        if self.use_qk_norm:
            new_qs = [self.q_norm(q) for q in new_qs]
            new_ks = [self.k_norm(k) for k in new_ks]

        new_qs = [self.rope(q, freqs_cos, freqs_sin) for q in new_qs]
        new_ks = [self.rope(k, freqs_cos, freqs_sin) for k in new_ks]
        all_ks = []
        all_vs = []
        for i in range(self.n_kv_heads):
            ks, out_cache_state = self.k_caches[i].update(
                new_ks[i], in_cache_state, out_cache_state
            )
            all_ks.append(ks)
            vs, out_cache_state = self.v_caches[i].update(
                new_vs[i], in_cache_state, out_cache_state
            )
            all_vs.append(vs)

        heads = []
        for i in range(self.n_heads):
            kv_idx = i // self.n_heads_per_kv_group
            attn = new_qs[i] @ all_ks[kv_idx].transpose(-2, -1)
            attn = attn * self.inv_scale
            attn = attn + mask
            attn = F.softmax(attn, dim=-1)
            heads.append(attn @ all_vs[kv_idx])

        y = torch.cat(heads, dim=-1)
        if self.use_conv2d:
            y = (
                self.wo(y.reshape(bsz, seq_len, 1, -1).transpose(1, 3))
                .transpose(1, 3)
                .reshape(bsz, seq_len, -1)
            )
        else:
            y = self.wo(y)
        return y, {"out_cache_state": out_cache_state}

    def load_weights_from_attention_mha(self, other: AttentionMHA):
        for i in range(self.n_heads):
            self.wqs[i].weight.data.copy_(
                other.wq.weight[i * self.head_dim : (i + 1) * self.head_dim, :]
            )

        for i in range(self.n_kv_heads):
            self.wks[i].weight.data.copy_(
                other.wk.weight[i * self.head_dim : (i + 1) * self.head_dim, :]
            )
            self.wvs[i].weight.data.copy_(
                other.wv.weight[i * self.head_dim : (i + 1) * self.head_dim, :]
            )

        self.wo.weight.data.copy_(other.wo.weight)

        if other.use_qk_norm:
            self.use_qk_norm = True
            self.q_norm = torch.nn.RMSNorm(other.q_norm_fn.dim, other.q_norm_fn.eps)
            self.q_norm.load_state_dict(other.q_norm_fn.state_dict())
            self.k_norm = torch.nn.RMSNorm(other.k_norm_fn.dim, other.k_norm_fn.eps)
            self.k_norm.load_state_dict(other.k_norm_fn.state_dict())

    def linear_to_conv2d(self):
        def transfer_weight(linear, conv2d):
            conv2d.weight.data.copy_(linear.weight[:, :, None, None])
            return conv2d

        self.wqs = nn.ModuleList(
            [
                transfer_weight(
                    linear,
                    nn.Conv2d(self.dim, self.head_dim, 1, bias=self.attention_qkv_bias),
                )
                for linear in self.wqs
            ]
        )
        self.wks = nn.ModuleList(
            [
                transfer_weight(
                    linear,
                    nn.Conv2d(self.dim, self.head_dim, 1, bias=self.attention_qkv_bias),
                )
                for linear in self.wks
            ]
        )
        self.wvs = nn.ModuleList(
            [
                transfer_weight(
                    linear,
                    nn.Conv2d(self.dim, self.head_dim, 1, bias=self.attention_qkv_bias),
                )
                for linear in self.wvs
            ]
        )
        self.wo = transfer_weight(
            self.wo,
            nn.Conv2d(
                self.n_heads * self.head_dim, self.dim, 1, bias=self.attention_qkv_bias
            ),
        )

        self.use_conv2d = True
