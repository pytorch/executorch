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

Three caches share the op: ``SequenceReferenceCache`` (one sequence),
``BatchedSequenceReferenceCache`` (many private sequence caches), and
``CellReferenceCache`` (many sequences over a shared pool of per-token cells,
with sharing and eviction). All store float KV.
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
    """One attention: what to attend over, which queries do it, how to mask it.

    A step is answered with a list of these -- one for a cache holding a single
    history, one per sequence for a cache holding a private history each. They
    cover the query axis in order, so ``q_len`` alone places each.
    """

    k: torch.Tensor  # [B, H_kv, total, head_dim] -- key history
    v: torch.Tensor  # [B, H_kv, total, v_head_dim] -- value history
    kind: MaskKind
    q_len: int  # query tokens this spec answers, following the one before it
    mask: Optional[torch.Tensor] = None  # EXPLICIT only: bool, true = attend


class LayerKind(Enum):
    FLAT = "flat"  # retains all history
    RING = "ring"  # sliding window over the newest `window` positions


@dataclass(frozen=True)
class LayerPolicy:
    """Per-layer cache kind and its parameters."""

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
class SequenceReferenceCache:
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

    def rewind(self, position: int) -> None:
        """Drop everything from ``position`` on, in every layer.

        A windowed layer retains only its last ``window`` positions, so it
        cannot go back further than that even though this reference keeps the
        older ones -- the window is applied to the mask here and to the storage
        in a byte layer, and a rewind past it would attend cells that layer no
        longer holds.
        """
        used = self._used[0]
        if position < 0 or position > used:
            raise ValueError(f"rewind to {position}: the history holds {used}")
        floor = max(
            (
                used - self.config.policy_for(layer_id).window
                for layer_id in range(self.config.n_layers)
                if self.config.policy_for(layer_id).window > 0
            ),
            default=0,
        )
        if position < floor:
            raise ValueError(
                f"rewind to {position}: a windowed layer retains only from {floor}"
            )
        for layer_id in range(self.config.n_layers):
            if self.config.sizing == CacheSizing.DYNAMIC:
                self._k[layer_id] = self._k[layer_id][:, :, :position, :]
                self._v[layer_id] = self._v[layer_id][:, :, :position, :]
            self._used[layer_id] = position

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
    ) -> List[AttendSpec]:
        """Place this step's K/V and return what to attend over.

        One sequence, one history, so the list is always one long.

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
            one AttendSpec over the whole history, ``total`` = prior length +
            q_len.
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

        return [self._spec(layer_id, k_hist, v_hist, q_len)]

    def _spec(
        self,
        layer_id: int,
        k_hist: torch.Tensor,
        v_hist: torch.Tensor,
        q_len: int,
    ) -> AttendSpec:
        """The mask semantic for q_len new cells at the tail of a total window.

        The new cells are at the tail, so query i attends keys up to
        ``i + total - q_len``, and a sliding window bounds it from below at
        ``i + total - q_len - window``. Whichever bound the fused kinds cannot
        express is what makes the step EXPLICIT.
        """
        total = k_hist.shape[-2]
        window = self.config.policy_for(layer_id).window
        windowed = 0 < window < total
        if q_len == 1 and not windowed:
            return AttendSpec(k=k_hist, v=v_hist, kind=MaskKind.NONE, q_len=q_len)
        if q_len == total and not windowed:
            return AttendSpec(k=k_hist, v=v_hist, kind=MaskKind.CAUSAL, q_len=q_len)
        # torch's is_causal is upper-left and expresses no window, so the band
        # is handed back explicitly.
        device = k_hist.device
        offsets = torch.arange(total, device=device) - torch.arange(
            q_len, device=device
        ).unsqueeze(-1)
        band = offsets <= total - q_len
        if windowed:
            band &= offsets > total - q_len - window
        return AttendSpec(
            k=k_hist, v=v_hist, kind=MaskKind.EXPLICIT, q_len=q_len, mask=band
        )


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
        ``seq_ids`` for ``declare_step``, and ``logits_indices`` selecting each
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


@dataclass(frozen=True)
class _SequenceSpan:
    seq_id: int
    start: int
    length: int


@experimental(
    "update_and_attend KV cache is experimental and may change without notice."
)
class BatchedSequenceReferenceCache:
    """A private ``SequenceReferenceCache`` per sequence in a flat batch.

    Projections share one model forward over the flattened token axis. Attention
    splits that axis into its declared sequence spans, runs independently over
    each sequence's private history, then concatenates the outputs in input
    order. No sequence attends another and no dense cross-sequence mask is built.
    """

    def __init__(self, config: CacheConfig):
        if config.batch_size != 1:
            raise ValueError(
                "batched sequence cache is flat on the token axis: batch_size must be 1"
            )
        self.config = config
        self._sequences: Dict[int, SequenceReferenceCache] = {}
        self._spans: List[_SequenceSpan] = []
        self._served: Set[int] = set()
        self._declared = False

    def declare_step(self, seq_ids: Sequence[int]) -> None:
        if not seq_ids:
            raise ValueError("a step carries at least one token")
        for seq_id in seq_ids:
            self._check_seq_id(seq_id)

        spans: List[_SequenceSpan] = []
        start = 0
        while start < len(seq_ids):
            seq_id = seq_ids[start]
            end = start + 1
            while end < len(seq_ids) and seq_ids[end] == seq_id:
                end += 1
            spans.append(_SequenceSpan(seq_id, start, end - start))
            start = end

        # capacity bounds the whole cache. Checked before anything is created so a refusal changes
        # nothing.
        held = sum(sequence.used(0) for sequence in self._sequences.values())
        if held + len(seq_ids) > self.config.capacity:
            raise RuntimeError(
                f"KV cache overflow: {held + len(seq_ids)} cells exceeds "
                f"capacity {self.config.capacity}"
            )

        for span in spans:
            if span.seq_id not in self._sequences:
                self._sequences[span.seq_id] = SequenceReferenceCache(self.config)

        self._spans = spans
        self._served.clear()
        self._declared = True

    def update_and_fetch(
        self,
        layer_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
        position: torch.Tensor,
    ) -> List[AttendSpec]:
        """Place each span's K/V in its own sequence and return one spec each.

        The specs follow the declared spans, so they cover the query axis in
        order and no sequence appears in another's window.
        """
        if not self._declared:
            raise RuntimeError(
                "no step declared: declare_step must precede every forward"
            )
        if layer_id in self._served:
            raise RuntimeError(
                f"layer {layer_id} served twice for one step: "
                "declare_step must precede every forward"
            )
        token_count = k.shape[-2]
        if not position.shape[0] == token_count == v.shape[-2]:
            raise ValueError("position, k, and v must have the same token count")
        if token_count != sum(span.length for span in self._spans):
            raise ValueError("the forward token count must match declare_step")
        if position.shape[-1] != 1:
            raise NotImplementedError(
                "sequence placement needs one position per token, got "
                f"{position.shape[-1]}"
            )
        self._check_positions(layer_id, position.reshape(-1).tolist())

        specs: List[AttendSpec] = []
        for span in self._spans:
            end = span.start + span.length
            sequence = self._sequences[span.seq_id]
            specs.extend(
                sequence.update_and_fetch(
                    layer_id,
                    k[:, :, span.start : end, :],
                    v[:, :, span.start : end, :],
                    position[span.start : end],
                )
            )

        self._served.add(layer_id)
        return specs

    def reset(self) -> None:
        self._sequences.clear()
        self._spans.clear()
        self._served.clear()
        self._declared = False

    def seq_rm(self, seq_id: int) -> None:
        """Release the whole sequence and its id."""
        self._check_seq_id(seq_id)
        self._sequences.pop(seq_id, None)
        self._invalidate_step()

    def rewind(self, seq_id: int, position: int) -> None:
        """Keep the sequence's ``[0, position)``, dropping the rest.

        A windowed layer has physically dropped what it no longer retains, so a
        target older than that is refused.
        """
        self._check_seq_id(seq_id)
        sequence = self._sequences.get(seq_id)
        if sequence is not None:
            sequence.rewind(position)
        self._invalidate_step()

    def _invalidate_step(self) -> None:
        self._spans.clear()
        self._served.clear()
        self._declared = False

    def max_seqs(self) -> Optional[int]:
        """None: a sequence is a dict entry, so only the cells they take bound them."""
        return None

    def pos(self, seq_id: int) -> int:
        """Where the sequence stands: one past its newest position."""
        self._check_seq_id(seq_id)
        sequence = self._sequences.get(seq_id)
        return sequence.used(0) if sequence is not None else 0

    def _check_positions(self, layer_id: int, positions: List[int]) -> None:
        """Every span continues its own sequence, from where that sequence ends.

        A private history appends at its used length and never reads
        ``position``, so a step that declared the wrong one would still place
        its tokens contiguously -- correct cells under the wrong names, and no
        later step would notice. A sequence spanned twice in one step continues
        across both.
        """
        ends: Dict[int, int] = {}
        for span in self._spans:
            at = ends.get(span.seq_id, self._sequences[span.seq_id].used(layer_id))
            got = positions[span.start : span.start + span.length]
            want = list(range(at, at + span.length))
            if got != want:
                raise ValueError(
                    f"sequence {span.seq_id} holds {at} positions on layer "
                    f"{layer_id}: the step declares {got}, not {want}"
                )
            ends[span.seq_id] = at + span.length

    @staticmethod
    def _check_seq_id(seq_id: int) -> None:
        # No upper bound: a sequence is a dict entry, not a bit in an owner set.
        if seq_id < 0:
            raise ValueError(f"seq_id must be non-negative, got {seq_id}")


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
    and sequence identity is supplied out-of-band. ``declare_step`` declares which
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
        self._declared = False  # set by declare_step, cleared by the step it authorizes
        self._plan: Optional[_CellStepPlan] = None
        self._served: Set[int] = set()

    # -- runner face: admission, lifecycle, sequence verbs ------------------

    def free_cells(self) -> int:
        return self._pos.count(-1)

    def _has_room(self, n: int = 1) -> bool:
        """Whether `n` more tokens fit: cache-wide, one cell per token.

        The bound is on cells, so a prefix shared by several sequences counts
        once and their lengths can sum past `capacity` while a step still fits.
        """
        return self.free_cells() >= n

    def pos(self, seq_id: int) -> int:
        """Where the sequence stands: one past its newest position.

        A sequence's cells are scattered across the pool, so the position lives
        on the cell and this scans for it.
        """
        self._check_seq_id(seq_id)
        bit = 1 << seq_id
        reached = -1
        for i, owners in enumerate(self._owners):
            if owners & bit:
                reached = max(reached, self._pos[i])
        return reached + 1

    def max_seqs(self) -> Optional[int]:
        """MAX_SEQS: one bit each in the owner bitset."""
        return MAX_SEQS

    def declare_step(self, seq_ids: Sequence[int]) -> None:
        """Declare the sequence each of the next forward's tokens belongs to.

        Admission is decided here, before the forward: the token count is known
        without the positions, and cells are interchangeable, so a step that
        passes this check cannot then fail to allocate.
        """
        if not seq_ids:
            raise ValueError("a step carries at least one token")
        for seq_id in seq_ids:
            self._check_seq_id(seq_id)
        if not self._has_room(len(seq_ids)):
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

    def seq_rm(self, seq_id: int) -> None:
        """Release the whole sequence and its id.

        A cell frees only once no sequence owns it, so a shared cell survives
        until its last owner lets go.
        """
        self._check_seq_id(seq_id)
        self._drop_from(seq_id, 0)

    def rewind(self, seq_id: int, position: int) -> None:
        """Keep the sequence's ``[0, position)``, dropping the rest.

        Always possible here: a windowed layer narrows the mask over cells that
        are still present, so no position is unrecoverable.
        """
        self._check_seq_id(seq_id)
        self._drop_from(seq_id, position)

    def _drop_from(self, seq_id: int, from_pos: int) -> None:
        bit = 1 << seq_id
        for i in range(self._used_end):
            if self._owners[i] & bit and self._pos[i] >= from_pos:
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
    ) -> List[AttendSpec]:
        """Scatter this step's K/V into its cells and return the read window.

        The first layer of a step allocates; the rest reuse that allocation, so
        the cells and the mask are computed once per forward, not once per
        layer. Every sequence reads the same window and the mask holds them
        apart, so the list is always one long. Args are as
        ``SequenceReferenceCache.update_and_fetch``.
        """
        if layer_id in self._served:
            raise RuntimeError(
                f"layer {layer_id} served twice for one step: "
                "declare_step must precede every forward"
            )
        if self._plan is None:
            self._plan = self._allocate(position)
        self._served.add(layer_id)

        read_len = self._plan.base.shape[-1]
        self._ensure(layer_id, read_len)
        cells = self._plan.cells
        self._k[layer_id][:, :, cells, :] = k.to(self.config.dtype)
        self._v[layer_id][:, :, cells, :] = v.to(self.config.dtype)
        return [
            AttendSpec(
                k=self._k[layer_id][:, :, :read_len, :],
                v=self._v[layer_id][:, :, :read_len, :],
                kind=MaskKind.EXPLICIT,
                q_len=len(cells),
                mask=self._plan.mask_for(self.config.policy_for(layer_id).window),
            )
        ]

    # -- internals ----------------------------------------------------------

    def _allocate(self, position: torch.Tensor) -> _CellStepPlan:
        # The plan indexes and masks the pools, so it is built where they live.
        device = self._k[0].device
        if not self._declared:
            raise RuntimeError(
                "no step declared: declare_step must precede every forward"
            )
        self._declared = False  # one declaration, one attempt at allocating it
        if position.shape[-1] != 1:
            raise NotImplementedError(
                f"cell placement needs one position per token, got {position.shape[-1]}"
            )
        positions = position.reshape(-1).tolist()
        if len(positions) != len(self._step_seq_ids):
            raise ValueError(
                f"declare_step declared {len(self._step_seq_ids)} tokens, "
                f"the forward carries {len(positions)}"
            )
        want = {seq_id: self.pos(seq_id) for seq_id in set(self._step_seq_ids)}
        for pos, seq_id in zip(positions, self._step_seq_ids):
            if pos != want[seq_id]:
                raise ValueError(
                    f"sequence {seq_id} continues at {want[seq_id]}, "
                    f"the step declares {pos}"
                )
            want[seq_id] += 1
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
        raise RuntimeError("no free cell")  # declare_step admitted the step

    def _shrink(self):
        while self._used_end > 0 and self._pos[self._used_end - 1] < 0:
            self._used_end -= 1

    def _invalidate_plan(self):
        # A mutated cell table leaves a built plan's cells and mask stale. The
        # step protocol state is deliberately left alone: a mutation must not
        # disguise a forward that skipped declare_step.
        self._plan = None

    @staticmethod
    def _check_seq_id(seq_id: int) -> None:
        # An id past the bitset silently makes owners a Python big-int, which
        # only surfaces much later as an int64 overflow building the mask.
        if not 0 <= seq_id < MAX_SEQS:
            raise ValueError(f"seq_id {seq_id} outside [0, {MAX_SEQS})")


def attend(
    q: torch.Tensor,
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
        q: ``[B, H_q, q_len, head_dim]`` -- queries (already RoPE-rotated), the
            ones this spec answers.
        spec: the K/V history to attend over and its mask semantic (NONE =
            attend all; CAUSAL = causal; EXPLICIT = the spec's bool mask).
        scale: attention softmax scale.
        out_dtype: output dtype.

    Returns:
        ``[B, H_q, q_len, v_head_dim]`` attention output, in ``out_dtype``.
    """
    k, v = spec.k, spec.v
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
