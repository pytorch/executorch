# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Eager reference (oracle) KV cache behind ``kvcache::update_and_attend``.

This is off-graph runtime state: it never appears in the exported graph, so the
physical sizing strategy is chosen here at construction time -- not baked into the
``.pte``. Two sizings are supported:

* ``STATIC``  -- preallocate a buffer of ``capacity`` cells; the used region
  advances within it (no realloc; models the static-shape backend constraint).
* ``DYNAMIC`` -- start empty and grow the used region lazily, up to ``capacity``.

Either way the cache bounds hard at ``capacity`` (required): memory grows lazily
but is capped, per the design's "grows lazily and bounds hard".

The cache places K/V and returns the history plus an ``AttendSpec`` (a mask *semantic*). The attend
mechanism (``attend`` below) is applied by the op/backend from that spec.

Scope for this initial slice: single sequence, contiguous placement, float KV.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import torch


class CacheSizing(Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"


class MaskKind(Enum):
    NONE = "none"  # decode: q_len == 1, one row sees all of history
    CAUSAL = "causal"  # prefill: q_len > 1, row i sees keys 0..offset+i


@dataclass
class AttendSpec:
    kind: MaskKind
    offset: int = 0  # cached tokens preceding this step


@dataclass
class CacheConfig:
    n_layers: int
    n_kv_heads: int
    head_dim: int
    capacity: int  # hard bound in cells; the cache never exceeds it
    sizing: CacheSizing = CacheSizing.DYNAMIC
    dtype: torch.dtype = torch.float32
    batch_size: int = 1

    def __post_init__(self):
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")


class ContiguousReferenceCache:
    """Per-layer contiguous float KV history for a single sequence."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self._k: List[torch.Tensor] = []
        self._v: List[torch.Tensor] = []
        self._used: List[int] = [0] * config.n_layers
        b, h, d = config.batch_size, config.n_kv_heads, config.head_dim
        init_len = config.capacity if config.sizing == CacheSizing.STATIC else 0
        for _ in range(config.n_layers):
            self._k.append(torch.zeros(b, h, init_len, d, dtype=config.dtype))
            self._v.append(torch.zeros(b, h, init_len, d, dtype=config.dtype))

    def used(self, layer_id: int) -> int:
        return self._used[layer_id]

    def reset(self):
        self._used = [0] * self.config.n_layers
        if self.config.sizing == CacheSizing.DYNAMIC:
            b, h, d = (
                self.config.batch_size,
                self.config.n_kv_heads,
                self.config.head_dim,
            )
            for i in range(self.config.n_layers):
                self._k[i] = torch.zeros(b, h, 0, d, dtype=self.config.dtype)
                self._v[i] = torch.zeros(b, h, 0, d, dtype=self.config.dtype)

    def update_and_fetch(
        self,
        layer_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
        position: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, AttendSpec]:
        """Place this step's K/V and return the full history + mask semantic.

        Per the design, ``position`` is the cache's placement + masking input.
        This contiguous single-sequence cache appends at its used length, so the
        causal offset is that prior length; non-contiguous (tree) caches will
        consume ``position`` directly to place and to build an Explicit mask.
        """
        q_len = k.shape[-2]
        used = self._used[layer_id]
        new_used = used + q_len
        cap = self.config.capacity
        if new_used > cap:
            raise RuntimeError(
                f"KV cache overflow on layer {layer_id}: "
                f"{new_used} cells exceeds capacity {cap}"
            )

        k = k.to(self.config.dtype)
        v = v.to(self.config.dtype)
        if self.config.sizing == CacheSizing.STATIC:
            self._k[layer_id][:, :, used:new_used, :] = k
            self._v[layer_id][:, :, used:new_used, :] = v
            k_hist = self._k[layer_id][:, :, :new_used, :]
            v_hist = self._v[layer_id][:, :, :new_used, :]
        else:
            self._k[layer_id] = torch.cat([self._k[layer_id], k], dim=2)
            self._v[layer_id] = torch.cat([self._v[layer_id], v], dim=2)
            k_hist = self._k[layer_id]
            v_hist = self._v[layer_id]
        self._used[layer_id] = new_used

        kind = MaskKind.NONE if q_len == 1 else MaskKind.CAUSAL
        return k_hist, v_hist, AttendSpec(kind=kind, offset=used)


def attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    spec: AttendSpec,
    scale: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Eager attend mechanism: apply SDPA to fetched K/V per the mask semantic."""
    n_q_heads = q.shape[1]
    n_kv_heads = k.shape[1]
    if n_q_heads != n_kv_heads:
        rep = n_q_heads // n_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    vf = v.to(torch.float32)

    scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale
    if spec.kind != MaskKind.NONE:
        q_len = qf.shape[2]
        total = kf.shape[2]
        query_pos = spec.offset + torch.arange(q_len, device=q.device)
        key_pos = torch.arange(total, device=q.device)
        visible = key_pos[None, :] <= query_pos[:, None]
        scores = scores.masked_fill(~visible[None, None], float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, vf)
    return out.to(out_dtype)
