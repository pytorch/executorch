# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cases for the attention path as the HF LLM exporter emits it: masks built
by comparing a key arange with the token positions, a static KV cache written
at those positions with index_copy_, and scaled_dot_product_attention under
causal, sliding-window and shared masks, with grouped KV heads."""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from executorch.backends.native.test.samples.cases import Case, Inputs
from torch.export import Dim


class _Fn(torch.nn.Module):
    def __init__(self, fn: Callable[..., object]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


def _positions(start: int, count: int) -> torch.Tensor:
    return torch.arange(start, start + count)


def _causal(positions: torch.Tensor, keys: int) -> torch.Tensor:
    """Key j is visible to the token at position p where j <= p, as the
    exporter's per-layer mask compares them."""
    return torch.arange(keys) <= positions.view(-1, 1)


def _sliding(positions: torch.Tensor, keys: int, window: int) -> torch.Tensor:
    """The causal mask times the window's lower bound, p - window < j: the
    product of comparisons Gemma's and Mistral's sliding layers export."""
    return _causal(positions, keys) * (
        (positions.view(-1, 1) - window) < torch.arange(keys)
    )


def _batched(mask: torch.Tensor) -> torch.Tensor:
    return mask[None, None, :, :].expand(1, -1, -1, -1)


class _Attention(torch.nn.Module):
    """SDPA of q [1, H, T, D] over k, v [1, Hkv, S, D] under a mask from the
    tokens' positions; `window` makes it a sliding-window mask."""

    def __init__(
        self,
        window: int | None = None,
        scale: float | None = None,
        repeat_kv: bool = False,
    ) -> None:
        super().__init__()
        self.window = window
        self.scale = scale
        self.repeat_kv = repeat_kv

    def forward(self, q, k, v, positions):
        keys = k.shape[2]
        mask = (
            _causal(positions, keys)
            if self.window is None
            else _sliding(positions, keys, self.window)
        )
        groups = q.shape[1] // k.shape[1]
        if self.repeat_kv and groups > 1:
            # transformers' repeat_kv, which SDPA then reads at full heads.
            k = k[:, :, None].expand(-1, -1, groups, -1, -1).flatten(1, 2)
            v = v[:, :, None].expand(-1, -1, groups, -1, -1).flatten(1, 2)
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=_batched(mask),
            scale=self.scale,
            enable_gqa=q.shape[1] != k.shape[1],
        )


class _Shared(torch.nn.Module):
    """A global and a sliding-window attention over one causal comparison, as
    a hybrid model's layers share their masks' values."""

    def __init__(self, window: int) -> None:
        super().__init__()
        self.window = window

    def forward(self, q, k, v, positions):
        keys = k.shape[2]
        causal = _causal(positions, keys)
        local = causal * ((positions.view(-1, 1) - self.window) < torch.arange(keys))
        gqa = q.shape[1] != k.shape[1]
        return (
            F.scaled_dot_product_attention(
                q, k, v, attn_mask=_batched(causal), enable_gqa=gqa
            ),
            F.scaled_dot_product_attention(
                q * 0.5, k, v, attn_mask=_batched(local), enable_gqa=gqa
            ),
        )


class _CachedAttention(torch.nn.Module):
    """The static KV cache path: write k and v at the tokens' positions into
    zero-initialized cache buffers with index_copy_, as the exporter's
    PositionedStaticLayer does, then attend over the whole cache."""

    def __init__(
        self, kv_heads: int, keys: int, dim: int, window: int | None = None
    ) -> None:
        super().__init__()
        self.attention = _Attention(window)
        cache = torch.zeros(1, kv_heads, keys, dim)
        self.register_buffer("key_cache", cache.clone(), persistent=False)
        self.register_buffer("value_cache", cache.clone(), persistent=False)

    def forward(self, q, k, v, positions):
        self.key_cache.index_copy_(2, positions, k)
        self.value_cache.index_copy_(2, positions, v)
        return self.attention(q, self.key_cache, self.value_cache, positions)


class _CacheWrite(torch.nn.Module):
    """index_copy_ of rows into a cache buffer at the tokens' positions,
    returning the whole cache."""

    def __init__(self, heads: int, keys: int, dim: int) -> None:
        super().__init__()
        self.register_buffer(
            "cache", torch.zeros(1, heads, keys, dim), persistent=False
        )

    def forward(self, x, positions):
        self.cache.index_copy_(2, positions, x)
        return self.cache * 1.0


def _qkv(
    heads: int,
    kv_heads: int,
    dim: int,
    tokens: int,
    keys: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(1, heads, tokens, dim, generator=generator)
    k = torch.randn(1, kv_heads, keys, dim, generator=generator)
    v = torch.randn(1, kv_heads, keys, dim, generator=generator)
    return q, k, v


def _attention_inputs(
    heads: int, kv_heads: int, dim: int, calls: list[tuple[int, int]], keys: int
) -> list[Inputs]:
    """Per (start, tokens): q of those tokens, k and v over all keys, and the
    tokens' positions."""
    generator = torch.Generator().manual_seed(0)
    return [
        (
            *_qkv(heads, kv_heads, dim, tokens, keys, generator),
            _positions(start, tokens),
        )
        for start, tokens in calls
    ]


def _cached_inputs(
    heads: int, kv_heads: int, dim: int, calls: list[tuple[int, int]]
) -> list[Inputs]:
    """Per (start, tokens): q, and the new k and v, of those tokens, and their
    positions."""
    generator = torch.Generator().manual_seed(0)
    return [
        (
            *_qkv(heads, kv_heads, dim, tokens, tokens, generator),
            _positions(start, tokens),
        )
        for start, tokens in calls
    ]


def _tokens(name: str = "tokens") -> Dim:
    return Dim(name, min=1, max=16)


def _prefill(name: str, model: torch.nn.Module, heads, kv_heads, dim, keys=8) -> Case:
    return Case(name, model, _attention_inputs(heads, kv_heads, dim, [(0, keys)], keys))


def _decode(
    name: str, model: torch.nn.Module, heads, kv_heads, dim, keys=64, at=(40, 63)
) -> Case:
    calls = [(p, 1) for p in at]
    return Case(name, model, _attention_inputs(heads, kv_heads, dim, calls, keys))


def sdpa_cases() -> list[Case]:
    tokens = _tokens()
    cases = [
        # Head dims 64 to 512 with two query heads per KV head.
        *(
            _prefill(f"causal_prefill_d{d}", _Attention(), 4, 2, d)
            for d in (64, 128, 256, 512)
        ),
        *(
            _decode(f"causal_decode_d{d}", _Attention(), 4, 2, d)
            for d in (64, 128, 256, 512)
        ),
        # Query heads per KV head: 1, 4 and 8.
        _prefill("causal_prefill_gqa1", _Attention(), 4, 4, 128),
        _prefill("causal_prefill_gqa4", _Attention(), 8, 2, 128),
        _decode("causal_decode_gqa8", _Attention(), 8, 1, 128),
        # transformers' repeat_kv before SDPA instead of grouped heads.
        _prefill("causal_prefill_repeat_kv", _Attention(repeat_kv=True), 8, 2, 64),
        # Gemma 4 12B's global layers: 16 query heads over one KV head of 512,
        # unscaled, and its sliding layers: 8 KV heads of 256.
        _decode("gemma_global_decode", _Attention(scale=1.0), 16, 1, 512),
        # ggml-cuda's head-512 kernel needs a mask over a multiple of 256 keys,
        # as a static cache has; over fewer, ggml runs it on the CPU.
        _decode(
            "gemma_global_decode_256_keys",
            _Attention(scale=1.0),
            16,
            1,
            512,
            keys=256,
            at=(200, 255),
        ),
        _prefill("gemma_global_prefill", _Attention(scale=1.0), 16, 1, 512),
        _decode("gemma_sliding_decode", _Attention(window=16, scale=1.0), 16, 8, 256),
        # A window of 3 over 8 tokens masks each row's oldest keys.
        _prefill("sliding_prefill", _Attention(window=3), 4, 2, 64),
        # Decode past the window: its keys start at p - 15.
        _decode("sliding_decode", _Attention(window=16), 4, 2, 128),
        _prefill("shared_masks_prefill", _Shared(window=3), 4, 2, 64),
        _decode("shared_masks_decode", _Shared(window=16), 8, 2, 128),
        # The first call stays mid-range so export traces a dynamic token count.
        Case(
            "dynamic_tokens",
            _Attention(),
            _attention_inputs(4, 2, 64, [(8, 8), (0, 1), (16, 16), (31, 1)], 32),
            dynamic_shapes=({2: tokens}, {}, {}, {0: tokens}),
        ),
    ]
    # Prefill, then decodes and a second prefill that write and read the cache
    # at an offset, crossing the window in the sliding case.
    kv_calls = [(0, 6), (6, 1), (7, 1), (8, 5), (13, 1)]
    for name, heads, kv_heads, dim, window in (
        ("dynamic_kv_cache_causal", 4, 2, 64, None),
        ("dynamic_kv_cache_sliding", 4, 2, 128, 4),
        ("dynamic_kv_cache_gemma_global", 16, 1, 512, None),
    ):
        keys_tokens = _tokens()
        cases.append(
            Case(
                name,
                _CachedAttention(kv_heads, 16, dim, window),
                _cached_inputs(heads, kv_heads, dim, kv_calls),
                dynamic_shapes=(
                    {2: keys_tokens},
                    {2: keys_tokens},
                    {2: keys_tokens},
                    {0: keys_tokens},
                ),
            )
        )
    return cases


def _write_inputs(heads: int, dim: int, calls: list[tuple[int, int]]) -> list[Inputs]:
    generator = torch.Generator().manual_seed(0)
    return [
        (
            torch.randn(1, heads, tokens, dim, generator=generator),
            _positions(start, tokens),
        )
        for start, tokens in calls
    ]


def index_put_cases() -> list[Case]:
    tokens = _tokens()
    write = _CacheWrite(2, 32, 64)
    return [
        Case("one_row_at_offset", write, _write_inputs(2, 64, [(17, 1)])),
        Case("rows_at_offset", write, _write_inputs(2, 64, [(9, 7)])),
        Case("last_rows", write, _write_inputs(2, 64, [(28, 4)])),
        # Writes accumulate in the cache across a chain's calls.
        Case(
            "dynamic_tokens",
            _CacheWrite(2, 32, 64),
            _write_inputs(2, 64, [(8, 8), (16, 1), (0, 8), (17, 15)]),
            dynamic_shapes=({2: tokens}, {0: tokens}),
        ),
    ]


def _mask_inputs(calls: list[tuple[int, int]], keys: int) -> list[Inputs]:
    """Per (start, tokens): a float [tokens, keys] the mask selects from, and
    the tokens' positions."""
    generator = torch.Generator().manual_seed(0)
    return [
        (torch.randn(tokens, keys, generator=generator), _positions(start, tokens))
        for start, tokens in calls
    ]


def _comparison(
    op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[[], list[Case]]:
    """Masks from `op` of a key arange and the tokens' positions, at an offset
    and far out, then `op` of floats with ties."""

    def cases() -> list[Case]:
        tokens = _tokens()
        far = 2**20

        def select(x, mask):
            return x * mask.to(x.dtype)

        def generator_ties() -> list[Inputs]:
            generator = torch.Generator().manual_seed(0)
            return [
                tuple(
                    torch.randint(-3, 4, (8, 64), generator=generator).float()
                    for _ in range(2)
                )
            ]

        return [
            # Key positions against a column of token positions.
            Case(
                "arange_positions",
                _Fn(lambda x, p: select(x, op(torch.arange(32), p.view(-1, 1)))),
                _mask_inputs([(9, 8), (0, 8), (24, 8)], 32),
            ),
            # A window's lower bound: positions minus a width against keys.
            Case(
                "window_bound",
                _Fn(lambda x, p: select(x, op(p.view(-1, 1) - 7, torch.arange(32)))),
                _mask_inputs([(9, 8), (20, 8)], 32),
            ),
            # A key arange at an offset, as a cache window's keys are.
            Case(
                "offset_arange",
                _Fn(lambda x, p: select(x, op(torch.arange(32) + 5, p.view(-1, 1)))),
                _mask_inputs([(9, 8)], 32),
            ),
            # Positions and keys past F16's exact integers.
            Case(
                "far_positions",
                _Fn(
                    lambda x, p: select(
                        x, op(torch.arange(32) + far, p.view(-1, 1) + far)
                    )
                ),
                _mask_inputs([(9, 8)], 32),
            ),
            Case(
                "floats",
                _Fn(lambda x, y: op(x, y).to(x.dtype)),
                generator_ties(),
            ),
            Case(
                "dynamic_tokens",
                _Fn(lambda x, p: select(x, op(torch.arange(32), p.view(-1, 1)))),
                _mask_inputs([(8, 8), (0, 1), (16, 16), (31, 1)], 32),
                dynamic_shapes=({0: tokens}, {0: tokens}),
            ),
        ]

    return cases


def le_cases() -> list[Case]:
    return _comparison(torch.le)()


def lt_cases() -> list[Case]:
    return _comparison(torch.lt)()


def ge_cases() -> list[Case]:
    return _comparison(torch.ge)()
