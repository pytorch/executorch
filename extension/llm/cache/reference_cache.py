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

Two caches share the op: ``ContiguousReferenceCache`` (one sequence appended in
place) and ``CellReferenceCache`` (many sequences over a pool of per-token cells,
with sharing and eviction). Both store float KV.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F

from executorch.exir._warnings import experimental


class CacheSizing(Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"


class MaskKind(Enum):
    NONE = "none"  # decode: q_len == 1, the single query sees all of history
    CAUSAL = "causal"  # prefill/continuation: query i sees keys up to its position
    EXPLICIT = "explicit"  # anything the other two cannot express: a mask tensor


@dataclass
class AttendSpec:
    kind: MaskKind
    mask: Optional[torch.Tensor] = None  # EXPLICIT only: bool, true = attend


class LayerKind(Enum):
    FLAT = "flat"  # retains all history
    RING = "ring"  # sliding window over the newest `window` positions


@dataclass(frozen=True)
class LayerPolicy:
    """Per-layer cache kind and its parameters. Mirrors the C++ LayerPolicy."""

    kind: LayerKind = LayerKind.FLAT
    window: int = 0  # RING only: window size in positions

    def __post_init__(self):
        if self.kind is LayerKind.RING and self.window <= 0:
            raise ValueError("a ring layer needs a positive window")
        if self.kind is LayerKind.FLAT and self.window != 0:
            raise ValueError("a flat layer retains all history; window must be 0")

    @classmethod
    def flat(cls) -> "LayerPolicy":
        return cls(kind=LayerKind.FLAT)

    @classmethod
    def ring(cls, window: int) -> "LayerPolicy":
        return cls(kind=LayerKind.RING, window=window)


@experimental(
    "update_and_attend KV cache is experimental and may change without notice."
)
@dataclass
class CacheConfig:
    n_layers: int
    n_kv_heads: int
    head_dim: int
    capacity: int  # hard bound in cells; the cache never exceeds it
    sizing: CacheSizing = CacheSizing.DYNAMIC
    dtype: torch.dtype = torch.float32
    batch_size: int = 1
    # Per-layer policy: one entry applies to every layer, else one per layer.
    layers: Sequence[LayerPolicy] = (LayerPolicy.flat(),)

    def __post_init__(self):
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if len(self.layers) not in (1, self.n_layers):
            raise ValueError("layers must be one policy, or one per layer")

    def policy_for(self, layer_id: int) -> LayerPolicy:
        return self.layers[0] if len(self.layers) == 1 else self.layers[layer_id]


@experimental(
    "update_and_attend KV cache is experimental and may change without notice."
)
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

        Args (BHSD):
            layer_id: which layer's history to update.
            k: ``[B, H_kv, q_len, head_dim]`` -- new keys for this step.
            v: ``[B, H_kv, q_len, v_head_dim]`` -- new values (``v_head_dim`` may
                differ from ``head_dim``, e.g. MLA).
            position: ``[q_len, n_dims]`` int -- per-query-token positions.

        Returns:
            ``(k_hist, v_hist, spec)`` -- history ``[B, H_kv, total, head_dim]`` /
            ``[B, H_kv, total, v_head_dim]`` (``total`` = prior length + q_len) and
            the AttendSpec mask semantic.
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

        return k_hist, v_hist, self._spec(layer_id, q_len, new_used, k.device)

    def _spec(
        self, layer_id: int, q_len: int, total: int, device: torch.device
    ) -> AttendSpec:
        """The mask semantic for q_len new cells at the tail of a total window.

        The new cells are at the tail, so query i attends keys up to
        ``i + total - q_len``, and a sliding window bounds it from below at
        ``i + total - q_len - window``. Whichever bound the fused kinds cannot
        express is what makes the step EXPLICIT.
        """
        window = self.config.policy_for(layer_id).window
        windowed = 0 < window < total
        if q_len == 1 and not windowed:
            return AttendSpec(kind=MaskKind.NONE)
        if q_len == total and not windowed:
            return AttendSpec(kind=MaskKind.CAUSAL)
        # A backend whose causal is lower-right aligned fuses the upper bound
        # (MLX does); torch's is_causal is upper-left, and neither expresses the
        # window, so the reference hands back the band itself.
        offsets = torch.arange(total, device=device) - torch.arange(
            q_len, device=device
        ).unsqueeze(-1)
        band = offsets <= total - q_len
        if windowed:
            band &= offsets > total - q_len - window
        return AttendSpec(kind=MaskKind.EXPLICIT, mask=band)


# A cell's owners are a bitset in a torch int64, so bit 63 (the sign bit) is out.
MAX_SEQS = 63


def flatten_step(
    sequences: Mapping[int, Tuple[torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor, List[int], torch.Tensor]:
    """Lay out one step's sequences on a single token axis.

    A step is flat: every sequence's tokens share one axis with B = 1, and the
    per-token arrays must stay aligned. Building them together is what keeps
    them so.

    It is a host helper, not part of the cache: the cache is handed only the
    sequence ids, and never sees the tokens themselves.

    Args:
        sequences: ``{seq_id: (tokens, start_pos)}`` -- each sequence's tokens
            with the token axis second (``[1, n]`` ids, or ``[1, n, hidden]``
            where the model takes embeddings), and the position its first
            token takes.

    Returns:
        ``(tokens, positions, seq_ids, logits_indices)`` -- tokens concatenated
        on the token axis and ``positions`` (``[n_tok, 1]``) as model inputs,
        ``seq_ids`` for ``begin_step``, and ``logits_indices`` selecting each
        sequence's last token, the rows worth running the LM head on.
    """
    tokens, positions, seq_ids, logits_indices = [], [], [], []
    for seq_id, (toks, start_pos) in sequences.items():
        tokens.append(toks)
        positions.extend(range(start_pos, start_pos + toks.shape[1]))
        seq_ids.extend([seq_id] * toks.shape[1])
        logits_indices.append(len(seq_ids) - 1)
    return (
        torch.cat(tokens, dim=1),
        torch.tensor(positions, dtype=torch.long).unsqueeze(-1),
        seq_ids,
        torch.tensor(logits_indices, dtype=torch.long),
    )


@dataclass
class _CellStepPlan:
    """One step's allocation, shared by every layer of that forward.

    Layers can differ in window, so the mask is per *policy* rather than per
    layer: membership and causality are common, and only the lower bound moves.
    `masks` memoizes one per distinct window (0 = unwindowed) as layers ask, so
    a mixed model costs two masks a step rather than one per layer.
    """

    cells: torch.Tensor  # [n_tok] long -- the cell each query token was given
    base: torch.Tensor  # [n_tok, read_len] bool -- occupied, same seq, causal
    cell_pos: torch.Tensor  # [read_len] -- the window's positions
    tok_pos: torch.Tensor  # [n_tok, 1] -- this step's positions
    masks: Dict[int, torch.Tensor]  # window -> mask; 0 is `base` itself

    def mask_for(self, window: int) -> torch.Tensor:
        if window not in self.masks:
            self.masks[window] = self.base & (self.cell_pos > self.tok_pos - window)
        return self.masks[window]


@experimental(
    "update_and_attend KV cache is experimental and may change without notice."
)
class CellReferenceCache:
    """Per-cell KV history for several sequences sharing one pool.

    Each cell holds one token's K/V plus that token's position and the set of
    sequences owning it, so a sequence need not be contiguous and two may share
    cells -- a fork sets a second bit instead of copying K/V. Visibility is then
    a property of the cell rather than of the layout: query i attends cell j iff
    j is occupied, shares a sequence with i, and is no newer than i -- and, on a
    windowed layer, no older than its window. No causal alignment can express
    that, so the spec is always EXPLICIT.

    The batch is flat: tokens from every sequence sit on one axis with B = 1,
    and sequence identity is supplied out-of-band. ``begin_step`` declares which
    sequence each of the next forward's tokens belongs to; the positions arrive
    with the forward itself, in the op's ``position`` tensor, so cells are
    allocated on the first layer of the step and memoized for the rest of it.

    Layers may window differently: the mask is memoized per window rather than
    per layer, over cells they all share. Nothing is evicted -- reclaiming under
    mixed windows needs one cache per policy group.

    DYNAMIC sizing grows the pool to the occupied extent, so a short session
    reserves a short pool rather than the whole context. Growth must keep every
    cell's index and its bytes -- a cell's index is its name, held by the plan
    and by ``_pos``/``_owners`` -- so it appends rows and never renumbers.
    """

    def __init__(self, config: CacheConfig):
        if config.batch_size != 1:
            raise ValueError(
                "cell cache is flat on the token axis: batch_size must be 1"
            )
        self.config = config
        cap = config.capacity
        self._pos: List[int] = [-1] * cap  # per cell; -1 = free
        self._owners: List[int] = [0] * cap  # per cell; owning-sequence bitset
        self._used_end = 0  # every occupied cell is in [0, used_end): the read window
        h, d = config.n_kv_heads, config.head_dim
        rows = cap if config.sizing == CacheSizing.STATIC else 0
        self._k = [
            torch.zeros(1, h, rows, d, dtype=config.dtype)
            for _ in range(config.n_layers)
        ]
        self._v = [
            torch.zeros(1, h, rows, d, dtype=config.dtype)
            for _ in range(config.n_layers)
        ]
        self._step_seq_ids: List[int] = []
        self._declared = False  # set by begin_step, cleared by the step it authorizes
        self._plan: Optional[_CellStepPlan] = None
        self._served: Set[int] = set()

    # -- runner face: admission, lifecycle, sequence verbs ------------------

    def free_cells(self) -> int:
        return self._pos.count(-1)

    def can_extend(self, n: int = 1) -> bool:
        """Whether `n` more tokens fit: cache-wide, one cell per token.

        The bound is on cells, so a prefix shared by several sequences counts
        once and their lengths can sum past `capacity` while a step still fits.
        """
        return self.free_cells() >= n

    def seq_len(self, seq_id: int) -> int:
        self._check_seq_id(seq_id)
        bit = 1 << seq_id
        return sum(1 for owners in self._owners if owners & bit)

    def begin_step(self, seq_ids: Sequence[int]) -> None:
        """Declare the sequence each of the next forward's tokens belongs to.

        Admission is decided here, before the forward: the token count is known
        without the positions, and cells are interchangeable, so a step that
        passes this check cannot then fail to allocate.
        """
        if not seq_ids:
            raise ValueError("a step carries at least one token")
        for seq_id in seq_ids:
            self._check_seq_id(seq_id)
        if not self.can_extend(len(seq_ids)):
            raise RuntimeError(
                f"KV cache full: {len(seq_ids)} tokens need as many cells, "
                f"{self.free_cells()} free"
            )
        self._step_seq_ids = list(seq_ids)
        self._declared = True
        self._plan = None
        self._served.clear()

    def seq_cp(self, src_id: int, dst_id: int, upto: Optional[int] = None) -> None:
        """Give dst_id a claim on src_id's cells -- a fork that copies no K/V.

        Shares src_id's cells at positions below `upto`; None shares all of them,
        forking at src_id's end. There is no lower bound: a shared cell keeps one
        position, so what can be shared is a prefix, not an arbitrary range.
        """
        self._check_seq_id(src_id)
        self._check_seq_id(dst_id)
        src_bit, dst_bit = 1 << src_id, 1 << dst_id
        for i in range(self._used_end):
            if self._owners[i] & src_bit and (upto is None or self._pos[i] < upto):
                self._owners[i] |= dst_bit
        self._invalidate_plan()

    def seq_rm(self, seq_id: int, p0: int = 0, p1: Optional[int] = None) -> None:
        """Drop seq_id's claim on positions [p0, p1); p1 = None runs to the end.

        A cell frees only once no sequence owns it, so removing a shared range
        reclaims nothing until the last owner lets go. seq_rm(s) drops the whole
        sequence, seq_rm(s, 0, k) evicts its oldest k positions, and seq_rm(s, k)
        truncates it at position k.
        """
        self._check_seq_id(seq_id)
        bit = 1 << seq_id
        for i in range(self._used_end):
            if self._owners[i] & bit and self._in_range(self._pos[i], p0, p1):
                self._owners[i] &= ~bit
                if self._owners[i] == 0:
                    self._pos[i] = -1
        self._shrink()
        self._invalidate_plan()

    def reset(self):
        self._pos = [-1] * self.config.capacity
        self._owners = [0] * self.config.capacity
        self._used_end = 0
        self._step_seq_ids = []
        self._declared = False
        self._plan = None
        self._served.clear()
        if self.config.sizing == CacheSizing.DYNAMIC:
            h, d = self.config.n_kv_heads, self.config.head_dim
            for i in range(self.config.n_layers):
                self._k[i] = torch.zeros(1, h, 0, d, dtype=self.config.dtype)
                self._v[i] = torch.zeros(1, h, 0, d, dtype=self.config.dtype)

    # -- op face ------------------------------------------------------------

    def update_and_fetch(
        self,
        layer_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
        position: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, AttendSpec]:
        """Scatter this step's K/V into its cells and return the read window.

        The first layer of a step allocates; the rest reuse that allocation, so
        the cells and the mask are computed once per forward, not once per
        layer. Args are as ``ContiguousReferenceCache.update_and_fetch``.
        """
        if layer_id in self._served:
            raise RuntimeError(
                f"layer {layer_id} served twice for one step: "
                "begin_step must precede every forward"
            )
        if self._plan is None:
            self._plan = self._allocate(position)
        self._served.add(layer_id)

        read_len = self._plan.base.shape[-1]
        self._ensure(layer_id, read_len)
        cells = self._plan.cells
        self._k[layer_id][:, :, cells, :] = k.to(self.config.dtype)
        self._v[layer_id][:, :, cells, :] = v.to(self.config.dtype)
        return (
            self._k[layer_id][:, :, :read_len, :],
            self._v[layer_id][:, :, :read_len, :],
            AttendSpec(
                kind=MaskKind.EXPLICIT,
                mask=self._plan.mask_for(self.config.policy_for(layer_id).window),
            ),
        )

    # -- internals ----------------------------------------------------------

    def _allocate(self, position: torch.Tensor) -> _CellStepPlan:
        # The plan indexes and masks the pools, so it is built where they live.
        device = self._k[0].device
        if not self._declared:
            raise RuntimeError(
                "no step declared: begin_step must precede every forward"
            )
        self._declared = False  # one declaration, one attempt at allocating it
        if position.shape[-1] != 1:
            raise NotImplementedError(
                "cell placement needs one position per token, got "
                f"{position.shape[-1]}"
            )
        positions = position.reshape(-1).tolist()
        if len(positions) != len(self._step_seq_ids):
            raise ValueError(
                f"begin_step declared {len(self._step_seq_ids)} tokens, "
                f"the forward carries {len(positions)}"
            )
        cells = [
            self._claim(pos, 1 << seq_id)
            for pos, seq_id in zip(positions, self._step_seq_ids)
        ]
        # Occupied, sharing a sequence, and no newer than the query -- plus, when
        # windowed, no older than its window. The step's own cells are already
        # placed, so a query sees itself and any earlier token of its sequence in
        # the same batch.
        n = self._used_end
        cell_pos = torch.tensor(self._pos[:n], device=device)
        cell_owners = torch.tensor(self._owners[:n], device=device)
        tok_pos = torch.tensor(positions, device=device).unsqueeze(-1)
        tok_bit = torch.tensor(
            [1 << seq_id for seq_id in self._step_seq_ids], device=device
        ).unsqueeze(-1)
        base = (cell_pos >= 0) & ((cell_owners & tok_bit) != 0) & (cell_pos <= tok_pos)
        return _CellStepPlan(
            cells=torch.tensor(cells, dtype=torch.long, device=device),
            base=base,
            cell_pos=cell_pos,
            tok_pos=tok_pos,
            masks={0: base},
        )

    def _ensure(self, layer_id: int, rows: int) -> None:
        """Make room for `rows` cells, doubling as the byte layer's pool does.

        Rows are appended, so a cell keeps the index it was claimed under and
        the K/V already stored there stays where the plan expects it.
        """
        have = self._k[layer_id].shape[2]
        if rows <= have:
            return
        grown = max(have, 1)
        while grown < rows:
            grown *= 2
        grown = min(grown, self.config.capacity)
        pad = torch.zeros(
            1,
            self.config.n_kv_heads,
            grown - have,
            self.config.head_dim,
            dtype=self.config.dtype,
        )
        self._k[layer_id] = torch.cat([self._k[layer_id], pad], dim=2)
        self._v[layer_id] = torch.cat([self._v[layer_id], pad.clone()], dim=2)

    def _claim(self, pos: int, owners: int) -> int:
        # Lowest free cell, which keeps the read window tight. The byte layer
        # keeps a free list rather than scanning.
        for i in range(self.config.capacity):
            if self._pos[i] < 0:
                self._pos[i] = pos
                self._owners[i] = owners
                self._used_end = max(self._used_end, i + 1)
                return i
        raise RuntimeError("no free cell")  # begin_step admitted the step

    def _shrink(self):
        while self._used_end > 0 and self._pos[self._used_end - 1] < 0:
            self._used_end -= 1

    def _invalidate_plan(self):
        # A mutated cell table leaves a built plan's cells and mask stale. The
        # step protocol state is deliberately left alone: a mutation must not
        # disguise a forward that skipped begin_step.
        self._plan = None

    @staticmethod
    def _check_seq_id(seq_id: int) -> None:
        # An id past the bitset silently makes owners a Python big-int, which
        # only surfaces much later as an int64 overflow building the mask.
        if not 0 <= seq_id < MAX_SEQS:
            raise ValueError(f"seq_id {seq_id} outside [0, {MAX_SEQS})")

    @staticmethod
    def _in_range(pos: int, p0: int, p1: Optional[int]) -> bool:
        return pos >= p0 and (p1 is None or pos < p1)


def attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    spec: AttendSpec,
    scale: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Eager attend mechanism: SDPA over fetched K/V per the mask semantic.

    Repeats K/V heads for GQA/MQA (``H_q`` a multiple of ``H_kv``), casts to fp32,
    and calls ``F.scaled_dot_product_attention`` -- one mechanism per kind:
    unmasked for NONE, ``is_causal`` for CAUSAL, the cache's own bool mask for
    EXPLICIT. CAUSAL is square-only here. The design's causal is lower-right
    aligned, which torch's upper-left ``is_causal`` matches only on a fresh full prefill,
    so a cache must declare EXPLICIT for a chunked or multi-turn step.

    Args (BHSD):
        q: ``[B, H_q, q_len, head_dim]`` -- queries (already RoPE-rotated).
        k: ``[B, H_kv, total, head_dim]`` -- key history.
        v: ``[B, H_kv, total, v_head_dim]`` -- value history.
        spec: mask semantic (NONE = attend all; CAUSAL = causal; EXPLICIT = the
            spec's bool mask).
        scale: attention softmax scale.
        out_dtype: output dtype.

    Returns:
        ``[B, H_q, q_len, v_head_dim]`` attention output, in ``out_dtype``.
    """
    n_q_heads = q.shape[1]
    n_kv_heads = k.shape[1]
    if n_q_heads != n_kv_heads:
        rep = n_q_heads // n_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    if spec.kind == MaskKind.CAUSAL and q.shape[-2] != k.shape[-2]:
        raise ValueError(
            "CAUSAL over a non-square window: torch's is_causal is upper-left "
            "aligned and would hide the prior cells. The cache must declare "
            "EXPLICIT with a lower-right band for a continuation."
        )

    out = F.scaled_dot_product_attention(
        q.to(torch.float32),
        k.to(torch.float32),
        v.to(torch.float32),
        attn_mask=spec.mask if spec.kind == MaskKind.EXPLICIT else None,
        is_causal=spec.kind == MaskKind.CAUSAL,
        scale=scale,
    )
    return out.to(out_dtype)
