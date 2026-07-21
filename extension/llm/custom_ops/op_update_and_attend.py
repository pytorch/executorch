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

from typing import Dict, Optional

import torch

from executorch.extension.llm.custom_ops.op_update_and_attend_reference import attend

_REGISTRY: Dict[str, object] = {}
_CURRENT_KEY: Optional[str] = None


def install_cache(cache_key: str, cache: object) -> None:
    _REGISTRY[cache_key] = cache


def uninstall_cache(cache_key: str) -> None:
    global _CURRENT_KEY
    _REGISTRY.pop(cache_key, None)
    if _CURRENT_KEY == cache_key:
        _CURRENT_KEY = None


def set_active(cache_key: str) -> None:
    """Select the active cache for the forward that follows (control channel)."""
    global _CURRENT_KEY
    if cache_key not in _REGISTRY:
        raise KeyError(f"no cache installed under key {cache_key!r}")
    _CURRENT_KEY = cache_key


def _current_cache() -> object:
    if _CURRENT_KEY is None:
        raise RuntimeError(
            "update_and_attend called with no active cache; "
            "call install_cache(...) then set_active(...)"
        )
    return _REGISTRY[_CURRENT_KEY]


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
    k_hist, v_hist, spec = _current_cache().update_and_fetch(layer_id, k, v, position)
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
    return torch.empty_like(q, dtype=out_dtype)
