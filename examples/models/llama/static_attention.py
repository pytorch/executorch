import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
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


logger = logging.getLogger(__name__)
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
                updated[..., -update_len:] = update[
                    ..., update_pos : update_pos + update_len
                ]
            else:
                update_len = update_len or update.size(-2)
                updated = torch.roll(cache, -update_len, -2)
                updated[..., -update_len:, :] = update[
                    ..., update_pos : update_pos + update_len, :
                ]

        if style == "smart_mask":
            updated = torch.clone(cache)
            if transpose:
                update_len = update_len or update.size(-1)
                updated[..., :, pos : pos + update_len] = update[
                    ..., :, update_pos : update_pos + update_len
                ]
            else:
                update_len = update_len or update.size(-2)
                updated[..., pos : pos + update_len, :] = update[
                    ..., update_pos : update_pos + update_len, :
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
    def __init__(
        self, input_len, cache_len, style, mask_val=float("-inf"), dtype=torch.float32
    ):
        self.input_len = input_len
        self.cache_len = cache_len
        assert style in ("shift_pointer", "smart_mask")
        self.style = style
        self.mask_val = mask_val
        self.unmasked_len = 0
        self.tensor = torch.zeros(1, input_len, input_len + cache_len, dtype=dtype)
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
    class NGramCache:
        def __init__(self, max_size):
            self.cache = deque()
            self.max_size = max_size

        def add(self, x):
            if x in self.cache:
                return
            if len(self.cache) == self.max_size:
                self.cache.popleft()
            self.cache.append(x)

        def __iter__(self):
            return iter(self.cache)

        def __str__(self):
            return str(self.cache)

    def __init__(
        self,
        config: ModelArgs,
        input_len: int,
        cache_len: int,
        dtype=torch.float32,
        style: str = "shift_pointer",
        mask_val: float = float("-inf"),
    ):
        self.mask = StaticAttentionMask(
            input_len, cache_len, style=style, mask_val=mask_val, dtype=dtype
        )

        rope = Rope(config)
        freqs = rope.get_freqs(None, config.max_seq_len)
        self.freqs_cos = freqs[0].to(dtype)
        self.freqs_sin = freqs[1].to(dtype)

        split_mha = config.attention_type in ("static", "static_shas")
        if split_mha:
            self.k_caches = {
                StaticKVCache.calculate_cache_key(layer_id, head_id): torch.zeros(
                    1, cache_len, config.head_dim, dtype=dtype
                )
                for layer_id in range(config.n_layers)
                for head_id in range(config.n_kv_heads)
            }
            self.v_caches = {
                StaticKVCache.calculate_cache_key(layer_id, head_id): torch.zeros(
                    1, cache_len, config.head_dim, dtype=dtype
                )
                for layer_id in range(config.n_layers)
                for head_id in range(config.n_kv_heads)
            }
        else:
            self.k_caches = {
                StaticKVCache.calculate_cache_key(layer_id, 0): torch.zeros(
                    1, config.n_kv_heads, cache_len, config.head_dim, dtype=dtype
                )
                for layer_id in range(config.n_layers)
            }
            self.v_caches = {
                StaticKVCache.calculate_cache_key(layer_id, 0): torch.zeros(
                    1, config.n_kv_heads, cache_len, config.head_dim, dtype=dtype
                )
                for layer_id in range(config.n_layers)
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
            new_tokens.append(y[:, :1, ...].argmax().item())
            if new_tokens[-1] in stop_tokens:
                break

        return new_tokens

    def lookahead_decode(  # noqa: C901
        self,
        model: Callable[..., Any],
        init_token: int,
        n: int,
        ngram_size: int,
        window_size: int,
        n_verifications: int,
        stop_tokens: Optional[List[int]] = None,
        ngram_caches: Optional[Dict[int, "StaticAttentionIOManager.NGramCache"]] = None,
    ):
        if self.cache_full:
            raise RuntimeError("KV cache is full.")

        if (ngram_size - 1) * (window_size + n_verifications) > self.input_len:
            raise RuntimeError(
                "Lookahead decoding setting not compatible with input length."
                f" input_len = {self.input_len},"
                f" ngram_size = {ngram_size},"
                f" window_size = {window_size},"
                f" n_verifications = {n_verifications}"
            )

        stop_tokens = stop_tokens or []
        if ngram_caches is None:
            ngram_caches = defaultdict(
                lambda: StaticAttentionIOManager.NGramCache(n_verifications)
            )

        self.mask.tensor[:, :, self.cache_len :] = self._get_lookahead_decoding_mask(
            ngram_size, window_size, n_verifications
        )
        logger.debug("Lookahead decoding mask: ")
        for i in range(self.input_len):
            logger.debug(
                " ".join(
                    ("X" if x == 0.0 else " ")
                    for x in self.mask.tensor[0][i][self.cache_len :]
                )
            )

        pos_offsets = self._get_lookahead_position_offsets(
            ngram_size, window_size, n_verifications
        )

        verification_offset = max(window_size * (ngram_size - 1), 1)
        new_tokens = [init_token]
        x = [init_token] * self.input_len
        inference_cnt = 0
        while len(new_tokens) < n + 1:
            # Update verification branch with cached n-grams.
            cache = ngram_caches[x[0]]
            for i, ngram in enumerate(cache):
                for j, token in enumerate(ngram):
                    x[verification_offset + i * (ngram_size - 1) + j] = token

            y, attn_updates = self._run_once(
                model,
                x,
                non_padded_len=1,
                freqs_cos_override=self.freqs_cos[pos_offsets + self.pos],
                freqs_sin_override=self.freqs_sin[pos_offsets + self.pos],
            )
            inference_cnt += 1
            # Only supports greedy decoding for now.
            y = y[0].argmax(dim=-1).tolist()
            new_tokens.append(y[0])
            logger.debug(f"{self.pos}: x = {x[0]}, y = {y[0]}")
            if new_tokens[-1] in stop_tokens:
                break

            # Collect new n-grams.
            for i in range(window_size):
                key = x[i]
                suffix = []
                for j in range(1, ngram_size - 1):
                    suffix.append(x[i + j * window_size])
                suffix.append(y[i + window_size * (ngram_size - 2)])
                ngram_caches[key].add(suffix)

            # Verification.
            longest_match = []
            matched_branch = None
            for i in range(n_verifications):
                match = [y[0]]
                j = 0
                # for j in range(ngram_size - 1):
                while (
                    j < ngram_size - 1
                    and x[verification_offset + (ngram_size - 1) * i + j] == match[-1]
                ):
                    match.append(y[verification_offset + (ngram_size - 1) * i + j])
                    j += 1
                if len(match) - 1 > len(longest_match):
                    longest_match = match[1:]
                    matched_branch = i

            if matched_branch is not None:
                logger.debug(
                    f"Matched {len(longest_match)} additional tokens from n-grams: {longest_match}"
                )
                for stop in stop_tokens:
                    if stop in longest_match:
                        longest_match = longest_match[: longest_match.index(stop) + 1]

                new_tokens.extend(longest_match)

                # Update KV caches and attention mask for the additional matched tokens.
                branch_offset = verification_offset + (ngram_size - 1) * matched_branch
                self._update_states(
                    attn_updates,
                    update_pos=branch_offset,
                    update_len=len(longest_match),
                )

            # Update lookahead branch.
            for i in range(ngram_size - 2):
                for j in range(window_size):
                    x[window_size * i + j] = x[window_size * (i + 1) + j]
            for j in range(window_size):
                x[window_size * (ngram_size - 2) + j] = y[
                    window_size * (ngram_size - 2) + j
                ]

            x[0] = new_tokens[-1]
            if new_tokens[-1] in stop_tokens:
                break

        logger.info(
            f"Generated {len(new_tokens) - 1} tokens with {inference_cnt} inference(s)."
        )
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

    def _get_lookahead_decoding_mask(
        self, ngram_size: int, window_size: int, n_verifications: int
    ) -> torch.Tensor:
        mask = torch.full((self.input_len, self.input_len), self.mask_val)
        mask[0][0] = 0.0

        lookahead_submask = torch.triu(
            torch.full((window_size, window_size), self.mask_val),
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
            torch.full((ngram_size - 1, ngram_size - 1), self.mask_val),
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
        # Input position offsets, used for indexing RoPE frequencies.
        pos_offsets = torch.zeros(self.input_len, dtype=torch.int32)
        idx = 0
        # Lookahead branches: [i + 0, i + 1, ..., i + window_size - 1] for time i.
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
            x_out = torch.stack([x_out_r, x_out_i], dim=-1).flatten(-2)
            return x_out


@register_attention("static")
class StaticAttention(Attention):
    """
    An attention implementation meant for NPUs that require static shapes and are not
    flexible with tensor operations needed to perform KV cache updates. MHA/GQA is
    implemented as multiple SHAs, and the KV caches keep valid data at the end so the
    model only needs to perform a concat to combine past and new data.
    """

    def __init__(
        self, config: ModelArgs, layer_id: int, rope: Rope, split_mha: bool = True
    ):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = (
            self.n_heads if config.n_kv_heads is None else config.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        assert self.n_heads % self.n_kv_heads == 0
        self.n_heads_per_kv_group = self.n_heads // self.n_kv_heads
        self.dim = config.dim
        self.head_dim = config.head_dim
        self.inv_scale = 1.0 / (float(self.head_dim) ** 0.5)
        self.attention_qkv_bias = config.attention_qkv_bias
        self.use_qk_norm = config.use_qk_norm
        self.qk_norm_before_rope = config.qk_norm_before_rope
        self.split_mha = split_mha
        self.use_conv2d = False

        if self.split_mha:
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
        else:
            self.wqs = nn.ModuleList(
                [
                    nn.Linear(
                        self.dim,
                        self.head_dim * self.n_heads,
                        bias=self.attention_qkv_bias,
                    )
                ]
            )
            self.wks = nn.ModuleList(
                [
                    nn.Linear(
                        self.dim,
                        self.head_dim * self.n_kv_heads,
                        bias=self.attention_qkv_bias,
                    )
                ]
            )
            self.wvs = nn.ModuleList(
                [
                    nn.Linear(
                        self.dim,
                        self.head_dim * self.n_kv_heads,
                        bias=self.attention_qkv_bias,
                    )
                ]
            )

            self.k_caches = nn.ModuleList([StaticKCache(layer_id, 0)])
            self.v_caches = nn.ModuleList([StaticVCache(layer_id, 0)])

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
        if (freqs_cos_override := kwargs.get("freqs_cos_override")) is not None:
            freqs_cos = freqs_cos_override  # pyre-ignore
        if (freqs_sin_override := kwargs.get("freqs_sin_override")) is not None:
            freqs_sin = freqs_sin_override  # pyre-ignore

        bsz, seq_len, dim = x.shape
        if self.use_conv2d:
            x = x.reshape(bsz, seq_len, 1, dim).transpose(1, 3)

        new_qs = [wq(x) for wq in self.wqs]
        new_ks = [wk(x) for wk in self.wks]
        new_vs = [wv(x) for wv in self.wvs]

        if self.use_conv2d:

            def from_conv2ds(ts):
                return [
                    t.reshape(bsz, self.head_dim, seq_len).transpose(1, 2) for t in ts
                ]

            new_qs = from_conv2ds(new_qs)
            new_ks = from_conv2ds(new_ks)
            new_vs = from_conv2ds(new_vs)

        if self.split_mha:
            y, out_cache_state = self._forward_sha(
                new_qs,
                new_ks,
                new_vs,
                freqs_cos,
                freqs_sin,
                **kwargs,
            )
        else:
            y, out_cache_state = self._forward_mha(
                new_qs[0],
                new_ks[0],
                new_vs[0],
                freqs_cos,
                freqs_sin,
                bsz,
                seq_len,
                **kwargs,
            )

        if self.use_conv2d:
            y = (
                self.wo(y.reshape(bsz, seq_len, 1, -1).transpose(1, 3))
                .transpose(1, 3)
                .reshape(bsz, seq_len, -1)
            )
        else:
            y = self.wo(y)

        return y, {"out_cache_state": out_cache_state}

    def _forward_sha(
        self,
        new_qs,
        new_ks,
        new_vs,
        freqs_cos,
        freqs_sin,
        **kwargs: ForwardOptions,
    ):
        mask = kwargs.get("mask")
        if (freqs_cos_override := kwargs.get("freqs_cos_override")) is not None:
            freqs_cos = freqs_cos_override  # pyre-ignore
        if (freqs_sin_override := kwargs.get("freqs_sin_override")) is not None:
            freqs_sin = freqs_sin_override  # pyre-ignore
        in_cache_state = kwargs.get("in_cache_state")
        out_cache_state = kwargs.get("out_cache_state")

        if self.use_qk_norm and self.qk_norm_before_rope:
            new_qs = [self.q_norm(q) for q in new_qs]
            new_ks = [self.k_norm(k) for k in new_ks]

        new_qs = [self.rope(q, freqs_cos, freqs_sin) for q in new_qs]
        new_ks = [self.rope(k, freqs_cos, freqs_sin) for k in new_ks]

        if self.use_qk_norm and not self.qk_norm_before_rope:
            new_qs = [self.q_norm(q) for q in new_qs]
            new_ks = [self.k_norm(k) for k in new_ks]

        all_ks = []
        all_vs = []
        for i in range(self.n_kv_heads if self.split_mha else 1):
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

        return torch.cat(heads, dim=-1), out_cache_state

    def _forward_mha(
        self,
        q,
        k,
        v,
        freqs_cos,
        freqs_sin,
        bsz,
        seq_len,
        **kwargs: ForwardOptions,
    ):
        mask = kwargs.get("mask")
        in_cache_state = kwargs.get("in_cache_state")
        out_cache_state = kwargs.get("out_cache_state")

        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm and self.qk_norm_before_rope:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = self.rope(q, freqs_cos, freqs_sin)
        k = self.rope(k, freqs_cos, freqs_sin)

        if self.use_qk_norm and not self.qk_norm_before_rope:
            q = self.q_norm(q)
            k = self.k_norm(k)

        k, out_cache_state = self.k_caches[0].update(k, in_cache_state, out_cache_state)
        v, out_cache_state = self.v_caches[0].update(v, in_cache_state, out_cache_state)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        return y.transpose(1, 2).contiguous().view(bsz, seq_len, -1), out_cache_state

    def load_weights_from_attention_mha(
        self, other: AttentionMHA, rms_norm_class=torch.nn.RMSNorm
    ):
        if self.split_mha:
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
        else:
            self.wqs[0].load_state_dict(other.wq.state_dict())
            self.wks[0].load_state_dict(other.wk.state_dict())
            self.wvs[0].load_state_dict(other.wv.state_dict())

        self.wo.weight.data.copy_(other.wo.weight)

        if other.use_qk_norm:
            self.use_qk_norm = True
            self.qk_norm_before_rope = other.qk_norm_before_rope
            self.q_norm = rms_norm_class(other.q_norm_fn.dim, other.q_norm_fn.eps).to(
                other.q_norm_fn.weight.dtype
            )
            self.q_norm.load_state_dict(other.q_norm_fn.state_dict())
            self.k_norm = rms_norm_class(other.k_norm_fn.dim, other.k_norm_fn.eps).to(
                other.k_norm_fn.weight.dtype
            )
            self.k_norm.load_state_dict(other.k_norm_fn.state_dict())

    def adopt_hf_rope(self):
        if self.rope.use_hf_rope:
            return

        if self.use_conv2d:
            raise RuntimeError(
                "adopt_hf_rope needs to be called before linear_to_conv2d"
            )

        # Permute weights of qk projections and norms to match HF RoPE's channel order.
        def permute(w, n_heads):
            shape = w.shape
            return (
                w.view((n_heads, -1, 2) + shape[1:])
                .transpose(1, 2)
                .reshape(shape)
                .contiguous()
            )

        for wq in self.wqs:
            wq.weight.data.copy_(
                permute(wq.weight.data, 1 if self.split_mha else self.n_heads)
            )

        for wk in self.wks:
            wk.weight.data.copy_(
                permute(wk.weight.data, 1 if self.split_mha else self.n_kv_heads)
            )

        if self.use_qk_norm:
            self.q_norm.weight.data.copy_(permute(self.q_norm.weight.data, 1))
            self.k_norm.weight.data.copy_(permute(self.k_norm.weight.data, 1))

        self.rope.use_hf_rope = True

    def linear_to_conv2d(self):
        if not self.split_mha:
            raise RuntimeError(
                "linear_to_conv2d is not supported when not splitting MHA"
            )
            return

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


@register_attention("static_mha")
class StaticAttentionMHA(StaticAttention):
    def __init__(self, config: ModelArgs, layer_id: int, rope: Rope):
        super().__init__(config, layer_id, rope, split_mha=False)
