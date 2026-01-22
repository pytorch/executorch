import copy
import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from executorch.examples.models.llama.attention import (
    Attention,
    AttentionMHA,
    ForwardOptions,
    register_attention,
)
from executorch.examples.models.llama.lora import LoRALinear
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.rope import Rope


logger = logging.getLogger(__name__)
_CacheMap = Dict[str, torch.Tensor]
# Key and value caches are kept separate so the key caches can be kept transposed.
_InputCacheState = Tuple[_CacheMap, _CacheMap]
_OutputCacheState = Tuple[_CacheMap, _CacheMap]


def none_throws(x: Optional[Any]) -> Any:
    assert x is not None
    return x


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
        seq_dim = -1 if transpose else -2
        cache_len = cache.size(seq_dim)
        if cache_len == 0:
            return
        if cache_len < update.size(seq_dim):
            update = torch.narrow(
                update,
                seq_dim,
                update.size(seq_dim) - cache_len,
                cache_len,
            )
            assert update.size(seq_dim) == cache_len

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
            available = cache.size(-2) - pos
            update_len = update_len or update.size(-1 if transpose else -2)
            if update_len > available:
                wrap = update_len - available
                update_len = available
            else:
                wrap = 0

            updated = torch.clone(cache)
            if transpose:
                updated[..., pos : pos + update_len] = update[
                    ..., update_pos : update_pos + update_len
                ]
                if wrap > 0:
                    update_pos += update_len
                    updated[..., :wrap] = update[..., update_pos : update_pos + wrap]

            else:
                updated[..., pos : pos + update_len, :] = update[
                    ..., update_pos : update_pos + update_len, :
                ]
                if wrap > 0:
                    update_pos += update_len
                    updated[..., :wrap, :] = update[
                        ..., update_pos : update_pos + wrap, :
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
        cache = in_cache_state[0].get(self.cache_key())
        if cache is None:
            return new_data, None
        if out_cache_state is None:
            out_cache_state = ({}, {})

        all_data = torch.cat([cache, new_data], dim=seq_dim)
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
        cache = in_cache_state[1].get(self.cache_key())
        if cache is None:
            return new_data, None
        if out_cache_state is None:
            out_cache_state = ({}, {})

        all_data = torch.cat([cache, new_data], dim=-2)
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

    def set_input_mask(self, input_mask):
        self.tensor[:, :, self.cache_len :] = input_mask

    def unmask(self, new_unmasked_len):
        if new_unmasked_len <= 0:
            return

        if self.style == "shift_pointer":
            self.tensor[
                :,
                :,
                max(
                    0, self.cache_len - self.unmasked_len - new_unmasked_len
                ) : self.cache_len
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
        config_or_model: Union[ModelArgs, nn.Module],
        input_len: int,
        cache_lens: Union[int, List[int]],
        batch_size: int = 1,
        dtype: torch.dtype = torch.float32,
        style: str = "shift_pointer",
        mask_val: float = float("-inf"),
    ):
        if isinstance(cache_lens, int):
            cache_lens_dict = defaultdict(lambda x=cache_lens: x)
            cache_lens = [cache_lens]
        else:
            cache_lens_dict = dict(enumerate(cache_lens))

        self._masks = {
            cl: StaticAttentionMask(
                input_len, cl, style=style, mask_val=mask_val, dtype=dtype
            )
            for cl in set(cache_lens)
        }

        if isinstance(config_or_model, ModelArgs):
            self._from_config(config_or_model, cache_lens_dict, batch_size, dtype)
        else:
            self._from_model(config_or_model, cache_lens_dict, batch_size, dtype)

        self.input_len = input_len
        self.style = style
        self.mask_val = mask_val
        self.pos = 0
        self.cache_full = False

    def _from_config(
        self,
        config: ModelArgs,
        cache_lens: Dict[int, int],
        batch_size: int,
        dtype: torch.dtype,
    ):
        rope = Rope(config)
        freqs = rope.get_freqs(None, config.max_context_len)
        self.freqs_cos = freqs[0].to(dtype)
        self.freqs_sin = freqs[1].to(dtype)

        split_mha = config.attention_type in ("static", "static_shas")
        if split_mha:
            self.k_caches = {
                StaticKVCache.calculate_cache_key(layer_id, head_id): torch.zeros(
                    batch_size,
                    cache_lens[layer_id],
                    none_throws(config.head_dim),
                    dtype=dtype,
                )
                for layer_id in range(config.n_layers)
                for head_id in range(none_throws(config.n_kv_heads))
                if cache_lens[layer_id] > 0
            }
            self.v_caches = {
                StaticKVCache.calculate_cache_key(layer_id, head_id): torch.zeros(
                    batch_size,
                    cache_lens[layer_id],
                    none_throws(config.head_dim),
                    dtype=dtype,
                )
                for layer_id in range(config.n_layers)
                for head_id in range(none_throws(config.n_kv_heads))
                if cache_lens[layer_id] > 0
            }
        else:
            self.k_caches = {
                StaticKVCache.calculate_cache_key(layer_id, 0): torch.zeros(
                    batch_size,
                    none_throws(config.n_kv_heads),
                    cache_lens[layer_id],
                    none_throws(config.head_dim),
                    dtype=dtype,
                )
                for layer_id in range(config.n_layers)
                if cache_lens[layer_id] > 0
            }
            self.v_caches = {
                StaticKVCache.calculate_cache_key(layer_id, 0): torch.zeros(
                    batch_size,
                    none_throws(config.n_kv_heads),
                    cache_lens[layer_id],
                    none_throws(config.head_dim),
                    dtype=dtype,
                )
                for layer_id in range(config.n_layers)
                if cache_lens[layer_id] > 0
            }

        self.generate_full_logits = config.generate_full_logits

    def _from_model(
        self,
        config: nn.Module,
        cache_lens: Dict[int, int],
        batch_size: int,
        dtype: torch.dtype,
    ):
        static_attentions = []
        for module in config.modules():
            if isinstance(module, StaticAttention):
                static_attentions.append(module)

        if not static_attentions:
            raise ValueError("No StaticAttention modules found in the provided module")

        config = copy.copy(static_attentions[0].rope.config)
        config.use_hf_rope = static_attentions[0].rope.use_hf_rope
        rope = Rope(config)
        freqs = rope.get_freqs(None, config.max_context_len)
        self.freqs_cos = freqs[0].to(dtype)
        self.freqs_sin = freqs[1].to(dtype)

        self.k_caches = {}
        self.v_caches = {}
        for attn in static_attentions:
            if attn.split_mha:
                for head_id in range(attn.n_heads):
                    cache_key = StaticKVCache.calculate_cache_key(
                        attn.layer_id, head_id
                    )
                    for cache in (self.k_caches, self.v_caches):
                        assert (
                            cache_key not in cache
                        ), "Found StaticAttention modules with duplicated layer_id"
                        cache[cache_key] = torch.zeros(
                            batch_size,
                            cache_lens[attn.layer_id],
                            attn.head_dim,
                            dtype=dtype,
                        )
            else:
                cache_key = StaticKVCache.calculate_cache_key(attn.layer_id, 0)
                for cache in (self.k_caches, self.v_caches):
                    assert (
                        cache_key not in cache
                    ), "Found StaticAttention modules with duplicated layer_id"
                    cache[cache_key] = torch.zeros(
                        batch_size,
                        attn.n_kv_heads,
                        cache_lens[attn.layer_id],
                        attn.head_dim,
                        dtype=dtype,
                    )

        self.generate_full_logits = True

    @property
    def masks(self):
        return {cache_len: mask.tensor for cache_len, mask in self._masks.items()}

    def reset(self):
        self.pos = 0
        self.cache_full = False
        for mask in self._masks.values():
            mask.reset()

    def prefill(
        self,
        model: Callable[..., Any],
        tokens: Union[List[int], torch.Tensor],
    ) -> torch.Tensor:
        if self.cache_full:
            raise RuntimeError("KV cache is full.")

        for mask in self._masks.values():
            mask.set_input_mask(
                torch.triu(
                    torch.full((1, self.input_len, self.input_len), self.mask_val),
                    diagonal=1,
                )
            )

        if isinstance(tokens, list):
            tokens = torch.tensor([tokens], dtype=torch.int32)

        logits = None
        all_logits = None
        for i in range(0, tokens.size(1), self.input_len):
            logits = self._run_once(model, tokens[:, i : i + self.input_len])[0]
            if self.generate_full_logits:
                if all_logits is None:
                    all_logits = logits
                else:
                    all_logits = torch.cat([all_logits, logits], dim=1)

        if self.generate_full_logits:
            return all_logits[:, : tokens.size(1), :]

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

        for mask in self._masks.values():
            mask.set_input_mask(
                torch.triu(
                    torch.full((1, self.input_len, self.input_len), self.mask_val),
                    diagonal=1,
                )
            )

        stop_tokens = stop_tokens or []
        new_tokens = [init_token]
        for _ in range(n):
            y = self._run_once(model, new_tokens[-1:])[0]
            if self.generate_full_logits:
                new_tokens.append(y[:, :1, ...].argmax().item())
            else:
                new_tokens.append(y.argmax().item())
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

        for mask in self._masks.values():
            mask.set_input_mask(
                self._get_lookahead_decoding_mask(
                    ngram_size, window_size, n_verifications
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
        tokens: Union[List[int], torch.Tensor],
        non_padded_len: Optional[int] = None,
        freqs_cos_override: Optional[torch.Tensor] = None,
        freqs_sin_override: Optional[torch.Tensor] = None,
    ):
        if isinstance(tokens, list):
            tokens = torch.tensor([tokens], dtype=torch.int32)
        n_tokens = tokens.size(1)
        if n_tokens < self.input_len:
            tokens = F.pad(tokens, (0, self.input_len - n_tokens))
        if freqs_cos_override is None:
            freqs_cos_override = self.freqs_cos[self.pos : self.pos + self.input_len]
        if freqs_sin_override is None:
            freqs_sin_override = self.freqs_sin[self.pos : self.pos + self.input_len]
        if not self.generate_full_logits:
            extra_attn_options = {
                "last_valid_token_pos": torch.tensor([n_tokens - 1], dtype=torch.long)
            }
        else:
            extra_attn_options = {}
        y, attn_updates = model(
            tokens,
            {
                "masks": self.masks,
                "freqs_cos_override": freqs_cos_override,
                "freqs_sin_override": freqs_sin_override,
                "in_cache_state": (self.k_caches, self.v_caches),
                **extra_attn_options,
            },
        )
        non_padded_len = non_padded_len or n_tokens
        self._update_states(attn_updates, 0, non_padded_len)

        return y, attn_updates

    def _update_states(self, attn_updates, update_pos, update_len):
        if attn_updates["out_cache_state"] is None:
            return
        for mask in self._masks.values():
            mask.unmask(update_len)
        k_cache_updates, v_cache_updates = attn_updates["out_cache_state"]
        for cache_id, update in k_cache_updates.items():
            self.k_caches[cache_id] = StaticKVCache.apply_update(
                self.k_caches[cache_id],
                update,
                self.pos,
                style=self.style,
                update_pos=update_pos,
                update_len=update_len,
            ).detach()
        for cache_id, update in v_cache_updates.items():
            self.v_caches[cache_id] = StaticKVCache.apply_update(
                self.v_caches[cache_id],
                update,
                self.pos,
                style=self.style,
                update_pos=update_pos,
                update_len=update_len,
            ).detach()
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
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.use_hf_rope = config.use_hf_rope

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
        self,
        config: ModelArgs,
        layer_id: int,
        rope: Rope,
        split_mha: bool = True,
        **kwargs: Any,
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
        self.enable_qnn_masked_softmax = kwargs.get("enable_qnn_masked_softmax", False)

        # This fixes numerics on iOS26 on Core ML
        # Possibly disable in future, depending on bug fixes in Core ML runtime
        self.decompose_sdpa_in_mha: bool = kwargs.get("decompose_sdpa_in_mha", False)

        # LoRA configuration
        self.target_modules = config.target_modules
        self.lora_rank = config.r
        self.lora_alpha = config.lora_alpha
        if self.target_modules:
            assert self.lora_rank is not None and self.lora_alpha is not None

        def _make_linear(in_dim: int, out_dim: int, bias: bool, lora_target: str) -> nn.Module:
            """Create a linear layer with optional LoRA support."""
            if self.target_modules is not None and lora_target in self.target_modules:
                # assert self.lora_rank is not None and self.lora_alpha is not None
                return LoRALinear(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    rank=self.lora_rank,
                    alpha=self.lora_alpha,
                    dropout=0.0,
                    use_bias=bias,
                )
            return nn.Linear(in_dim, out_dim, bias=bias)

        if self.split_mha:
            self.wqs = nn.ModuleList(
                [
                    _make_linear(self.dim, self.head_dim, self.attention_qkv_bias, "q_proj")
                    for _ in range(self.n_heads)
                ]
            )
            self.wks = nn.ModuleList(
                [
                    _make_linear(self.dim, self.head_dim, self.attention_qkv_bias, "k_proj")
                    for _ in range(self.n_kv_heads)
                ]
            )
            self.wvs = nn.ModuleList(
                [
                    _make_linear(self.dim, self.head_dim, self.attention_qkv_bias, "v_proj")
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
                    _make_linear(
                        self.dim,
                        self.head_dim * self.n_heads,
                        self.attention_qkv_bias,
                        "q_proj",
                    )
                ]
            )
            self.wks = nn.ModuleList(
                [
                    _make_linear(
                        self.dim,
                        self.head_dim * self.n_kv_heads,
                        self.attention_qkv_bias,
                        "k_proj",
                    )
                ]
            )
            self.wvs = nn.ModuleList(
                [
                    _make_linear(
                        self.dim,
                        self.head_dim * self.n_kv_heads,
                        self.attention_qkv_bias,
                        "v_proj",
                    )
                ]
            )

            self.k_caches = nn.ModuleList([StaticKCache(layer_id, 0)])
            self.v_caches = nn.ModuleList([StaticVCache(layer_id, 0)])

        self.wo = _make_linear(self.n_heads * self.head_dim, self.dim, False, "o_proj")
        self.rope = _Rope(rope.params)
        self.layer_id = layer_id

        if self.use_qk_norm:
            self.q_norm = torch.nn.RMSNorm(self.head_dim, config.norm_eps)
            self.k_norm = torch.nn.RMSNorm(self.head_dim, config.norm_eps)
        else:
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()

    @classmethod
    def from_attention_mha(
        cls,
        other: AttentionMHA,
        split_mha: bool = True,
        rms_norm_class=torch.nn.RMSNorm,
        **kwargs: Any,
    ) -> "StaticAttention":
        config = ModelArgs(
            dim=other.dim,
            n_layers=1,  # Not used in attention layer
            n_heads=other.n_heads,
            n_kv_heads=other.n_kv_heads,
            head_dim=other.head_dim,
            max_batch_size=other.max_batch_size,
            max_context_len=other.max_context_len,
            attention_qkv_bias=other.attention_qkv_bias,
            use_qk_norm=other.use_qk_norm,
            qk_norm_before_rope=other.qk_norm_before_rope,
            norm_eps=other.q_norm_fn.eps if other.use_qk_norm else 1e-5,
        )

        instance = cls(
            config=config,
            layer_id=other.layer_id,
            rope=other.rope,
            split_mha=split_mha,
            **kwargs,
        )
        instance.load_weights_from_attention_mha(other, rms_norm_class=rms_norm_class)

        return instance

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
            x = x.reshape(bsz, -1, 1, dim).transpose(1, 3)

        new_qs = [wq(x) for wq in self.wqs]
        new_ks = [wk(x) for wk in self.wks]
        new_vs = [wv(x) for wv in self.wvs]

        if self.use_conv2d:

            def from_conv2ds(ts):
                return [t.reshape(bsz, self.head_dim, -1).transpose(1, 2) for t in ts]

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
                seq_len,
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
                self.wo(
                    y.reshape(bsz, -1, 1, self.n_heads * self.head_dim).transpose(1, 3)
                )
                .transpose(1, 3)
                .reshape(bsz, -1, self.dim)
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
        seq_len,
        **kwargs: ForwardOptions,
    ):
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

        cache_len = all_ks[0].size(-2) - seq_len
        mask = kwargs["masks"][cache_len]

        heads = []
        for i in range(self.n_heads):
            kv_idx = i // self.n_heads_per_kv_group
            attn = new_qs[i] @ all_ks[kv_idx].transpose(-2, -1)
            attn = attn * self.inv_scale
            if self.enable_qnn_masked_softmax:
                attn_min = torch.amin(attn, dim=-1, keepdim=True)
                minus_value = -20
                attn = torch.where(
                    mask == 0, attn, attn_min + minus_value
                )  # prye-ignore
            else:
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

        mask = None
        masks = kwargs.get("masks")
        if masks:
            cache_len = k.size(-2) - seq_len
            mask = masks[cache_len]

        if not self.decompose_sdpa_in_mha:
            if self.n_rep > 1:
                k = k.repeat_interleave(self.n_rep, dim=1)
                v = v.repeat_interleave(self.n_rep, dim=1)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            # We remove bsz dim to keep matmul's on 4D tensors
            # Core ML sometimes fails at runtime when given 5D tensors
            assert bsz == 1, "Batch size > 1 not supported yet"

            n_kv = self.n_kv_heads
            n_rep = self.n_rep
            D = self.head_dim

            # Explicitly track lengths; they are NOT necessarily equal.
            Tq = q.size(-2)  # query length (current step/window), e.g. 64
            Tk = k.size(-2)  # key/value length (cache length), e.g. 2048

            # Group Q to match KV layout
            # q: (bsz=1, n_heads, Tq, D), with n_heads = n_kv * n_rep
            # 1 * n_heads * Tq * D == n_kv * n_rep * Tq * D
            # q_grouped: (n_kv, n_rep, Tq, D)
            q_grouped = q.view(n_kv, n_rep, Tq, D)

            # Prepare K for grouped KV matmul
            # k: (1, n_kv, Tk, d) -> (n_kv, 1, Tk, D)
            k_grouped = k.view(n_kv, 1, Tk, D)

            # (n_kv, n_rep, Tq, Tk)
            attn_grouped = q_grouped @ k_grouped.transpose(-2, -1)
            attn_grouped = attn_grouped * self.inv_scale

            # Ungroup, add mask, and regroup
            attn_grouped = attn_grouped.view(1, self.n_heads, Tq, Tk)
            attn_grouped = attn_grouped + mask
            attn_grouped = F.softmax(attn_grouped, dim=-1)
            attn_grouped = attn_grouped.view(n_kv, n_rep, Tq, Tk)

            # Group v
            v_grouped = v.view(n_kv, 1, Tk, D)
            y_grouped = attn_grouped @ v_grouped

            # Ungroup y
            y = y_grouped.view(1, self.n_heads, Tq, D)

        return y.transpose(1, 2).contiguous().view(bsz, seq_len, -1), out_cache_state

    def load_weights_from_attention_mha(
        self, other: AttentionMHA, rms_norm_class=torch.nn.RMSNorm
    ):
        if self.split_mha:
            for i in range(self.n_heads):
                self.wqs[i].weight.data.copy_(
                    # pyre-ignore[29]
                    other.wq.weight[i * self.head_dim : (i + 1) * self.head_dim, :]
                )

            for i in range(self.n_kv_heads):
                self.wks[i].weight.data.copy_(
                    # pyre-ignore[29]
                    other.wk.weight[i * self.head_dim : (i + 1) * self.head_dim, :]
                )
                self.wvs[i].weight.data.copy_(
                    # pyre-ignore[29]
                    other.wv.weight[i * self.head_dim : (i + 1) * self.head_dim, :]
                )
        else:
            self.wqs[0].load_state_dict(other.wq.state_dict())
            self.wks[0].load_state_dict(other.wk.state_dict())
            self.wvs[0].load_state_dict(other.wv.state_dict())

        self.wo.weight.data.copy_(other.wo.weight)  # pyre-ignore[6]

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
    def __init__(self, config: ModelArgs, layer_id: int, rope: Rope, **kwargs: Any):
        super().__init__(config, layer_id, rope, split_mha=False, **kwargs)


def transform_attention_mha_to_static_attention(
    model: nn.Module,
    split_mha: bool = True,
    inplace: bool = True,
    use_conv2d: bool = False,
    use_hf_rope: bool = False,
    **kwargs: Any,
) -> nn.Module:
    if not inplace:
        import copy

        model = copy.deepcopy(model)

    def helper(m):
        for name, child in list(m.named_children()):
            if isinstance(child, AttentionMHA):
                static_attn = StaticAttention.from_attention_mha(
                    child, split_mha=split_mha, **kwargs
                )
                # Note: HF RoPE needs to be applied before linear to conv2d
                if use_hf_rope:
                    static_attn.adopt_hf_rope()
                if use_conv2d:
                    static_attn.linear_to_conv2d()

                setattr(m, name, static_attn)
            else:
                helper(child)

        return m

    return helper(model)
