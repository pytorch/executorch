# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The neutral ``kvcache::update_and_attend`` custom op.

Functional to the tracer (``mutates_args=()``): the KV cache is *not* a graph
tensor but off-graph runtime state, reached through a process-global registry.
When the exported program is run eagerly, the real kernel dispatches to
the installed cache via ``layer_id``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, Optional

import torch

from executorch.extension.llm.cache.reference_cache import attend


class CacheRegistry:
    """Off-graph KV caches keyed by cache_key, with one active at a time."""

    def __init__(self) -> None:
        self._caches: Dict[str, object] = {}
        self._active: Optional[str] = None

    def install(self, cache_key: str, cache: object) -> None:
        self._caches[cache_key] = cache

    def uninstall(self, cache_key: str) -> None:
        self._caches.pop(cache_key, None)
        if self._active == cache_key:
            self._active = None

    @contextmanager
    def active(self, cache_key: str) -> Iterator[None]:
        """Select the active cache for the enclosed forward(s) (control channel)."""
        if cache_key not in self._caches:
            raise KeyError(f"no cache installed under key {cache_key!r}")
        prev = self._active
        self._active = cache_key
        try:
            yield
        finally:
            self._active = prev

    def current(self) -> object:
        if self._active is None:
            raise RuntimeError(
                "update_and_attend called with no active cache; "
                "install a cache and enter its active(...) scope"
            )
        return self._caches[self._active]


REGISTRY = CacheRegistry()


@torch.library.custom_op("kvcache::update_and_attend", mutates_args=())
def update_and_attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    position: torch.Tensor,
    layer_id: int,
    scale: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Append this step's k/v to the layer's cache, then attend q over history.

    This op and its cache API are experimental and may change without notice.

    Tensors are BHSD (batch, heads, seq, dim); GQA/MQA is handled natively
    (``H_q`` is a multiple of ``H_kv``). q/k are already RoPE-rotated.

    Args:
        q: ``[B, H_q, q_len, head_dim]`` -- queries for this step's tokens.
        k: ``[B, H_kv, q_len, head_dim]`` -- new keys; ``H_kv <= H_q`` (GQA),
            same ``head_dim`` as q.
        v: ``[B, H_kv, q_len, v_head_dim]`` -- new values; ``v_head_dim`` may
            differ from ``head_dim`` (e.g. MLA).
        position: ``[q_len, n_dims]`` int -- per-query-token positions for cache
            placement + masking; ``n_dims`` = position components per token (1
            for standard sequence positions).
        layer_id: which layer's cache to use (node constant).
        scale: attention softmax scale (node constant).
        out_dtype: output dtype, independent of KV storage precision (node
            constant).

    Returns:
        ``[B, H_q, q_len, v_head_dim]`` attention output, in ``out_dtype``.
    """
    k_hist, v_hist, spec = REGISTRY.current().update_and_fetch(layer_id, k, v, position)
    return attend(q, k_hist, v_hist, spec, scale, out_dtype)


@update_and_attend.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    position: torch.Tensor,
    layer_id: int,
    scale: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return q.new_empty((*q.shape[:-1], v.shape[-1]), dtype=out_dtype)
