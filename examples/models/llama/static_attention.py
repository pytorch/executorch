from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

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
    def apply_update(cache, update, transpose=False):
        """
        After inference, update the cache state for next iteration. The runtime needs to
        implement the same operation.
        """
        if transpose:
            update_len = update.size(-1)
            updated = torch.roll(cache, -update_len, -1)
            updated[:, :, -update_len:] = update
        else:
            update_len = update.size(-2)
            updated = torch.roll(cache, -update_len, -2)
            updated[:, -update_len:, :] = update

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


def _apply_rotary_embedding(
    x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> torch.Tensor:
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

        self.wqs = nn.ModuleList(
            [
                nn.Linear(self.dim, self.head_dim, bias=False)
                for _ in range(self.n_heads)
            ]
        )
        self.wks = nn.ModuleList(
            [
                nn.Linear(self.dim, self.head_dim, bias=False)
                for _ in range(self.n_kv_heads)
            ]
        )
        self.wvs = nn.ModuleList(
            [
                nn.Linear(self.dim, self.head_dim, bias=False)
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

        new_qs = [self.wqs[i](x) for i in range(self.n_heads)]
        new_ks = [self.wks[i](x) for i in range(self.n_kv_heads)]
        new_vs = [self.wvs[i](x) for i in range(self.n_kv_heads)]
        new_qs = [_apply_rotary_embedding(q, freqs_cos, freqs_sin) for q in new_qs]
        new_ks = [_apply_rotary_embedding(k, freqs_cos, freqs_sin) for k in new_ks]

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
            attn = attn + mask  # pyre-ignore
            attn = F.softmax(attn, dim=-1)
            heads.append(attn @ all_vs[kv_idx])

        y = torch.cat(heads, dim=-1)
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
