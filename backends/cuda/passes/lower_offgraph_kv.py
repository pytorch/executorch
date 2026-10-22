# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from typing import Any

import torch

# Imported for its registration side effect: the decomposition calls
# torch.ops.triton.sdpa, which only exists once this module has been loaded.
from executorch.backends.cuda.triton.kernels.sdpa import sdpa  # noqa: F401
from executorch.backends.cuda.triton.kernels.sdpa_varlen import (  # noqa: F401
    sdpa_varlen,
)
from executorch.backends.transforms.utils import create_constant_placeholder
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind
from torch.fx.experimental.proxy_tensor import make_fx


OFFGRAPH_KV_COMPILE_SPEC = "offgraph_kv_manifest"
# Consumed by the runtime, not by this pass: "input_index:dim" naming
# the input whose extent is the number of tokens a step writes.
OFFGRAPH_KV_STEP_WIDTH_COMPILE_SPEC = "offgraph_kv_step_width"
OFFGRAPH_KV_FQN_PREFIX = "__et_offgraph_kv_"

# Cell layout: every sequence of a batch shares one pool of per-token cells,
# and the runtime cache writes the step's placement and visibility into these
# buffers before each forward. Each is declared once per program at a fixed
# shape, so the runtime binds it at a fixed address that a captured CUDA graph
# may keep.
OFFGRAPH_KV_CELLS_FQN = OFFGRAPH_KV_FQN_PREFIX + "cells"
OFFGRAPH_KV_READ_LEN_FQN = OFFGRAPH_KV_FQN_PREFIX + "read_len"


# Batched-sequence layout: each sequence keeps its own column of every layer's
# storage, and the runtime cache writes which column each token belongs to
# (seq_cols) and each run of one sequence's query rows (seq_info) before each
# forward. Declared once per program at a fixed shape, like the cell buffers.
OFFGRAPH_KV_SEQ_COLS_FQN = OFFGRAPH_KV_FQN_PREFIX + "seq_cols"
OFFGRAPH_KV_SEQ_INFO_FQN = OFFGRAPH_KV_FQN_PREFIX + "seq_info"

# Layouts whose step packs tokens of several sequences on one axis.
PACKED_LAYOUTS = ("cell", "batched-sequence")


def batched_sequence_max_segments(max_seqs: int) -> int:
    """Rows of seq_info: one per run of a sequence's query rows in a step.

    Same rule as batched_sequence_max_segments() in
    backends/cuda/runtime/cuda_kv_cache.h, which checks steps against it.
    """
    return 2 * max_seqs


def offgraph_kv_mask_fqn(window: int) -> str:
    """The step mask shared by every layer of ``window`` (0 = full history)."""
    return f"{OFFGRAPH_KV_FQN_PREFIX}mask_w{window}"


def ring_physical_capacity(window: int, max_write: int) -> int:
    """Slots a ring layer needs to serve one step of up to ``max_write`` tokens.

    A step writes all its tokens before attending, and its earliest query still
    reads back to ``position - window + 1``, so ``window + max_write - 1``
    positions must be live at once. Sizing the ring to the window alone lets a
    step overwrite cells its own earlier queries still attend to.

    Matches ``RingPolicy`` in executorch/extension/llm/cache/sequence_cache.h;
    the CUDA runtime applies the same formula when it allocates.
    """
    if window <= 0:
        raise ValueError("ring cache requires a positive window")
    if max_write <= 0:
        raise ValueError("ring cache requires a positive max_write")
    return window + max_write - 1


def parse_offgraph_kv_manifest(value: bytes) -> dict[str, Any]:  # noqa: C901
    manifest = json.loads(value.decode("utf-8"))
    if manifest.get("version") != 1:
        raise ValueError("off-graph KV manifest version must be 1")
    maximum_capacity = manifest.get("maximum_capacity")
    if not isinstance(maximum_capacity, int) or maximum_capacity <= 0:
        raise ValueError("off-graph maximum_capacity must be positive")
    max_write = manifest.get("max_write")
    if not isinstance(max_write, int) or max_write <= 0:
        raise ValueError("off-graph max_write must be positive")
    layout = manifest.setdefault("layout", "sequence")
    if layout not in ("sequence", *PACKED_LAYOUTS):
        raise ValueError(f"invalid off-graph KV layout {layout!r}")
    if layout == "batched-sequence":
        for key in ("max_seqs", "max_segments"):
            if not isinstance(manifest.get(key), int) or manifest[key] <= 0:
                raise ValueError(f"off-graph batched-sequence layout needs {key} > 0")
        if manifest["max_segments"] != batched_sequence_max_segments(
            manifest["max_seqs"]
        ):
            raise ValueError(
                "off-graph batched-sequence max_segments must equal "
                "batched_sequence_max_segments(max_seqs)"
            )
        if not isinstance(manifest.setdefault("causal", True), bool):
            raise ValueError("off-graph causal must be a bool")
        if not isinstance(manifest.setdefault("context_window", False), bool):
            raise ValueError("off-graph context_window must be a bool")
        if manifest["context_window"] and manifest["causal"]:
            raise ValueError("off-graph context_window requires non-causal attention")
    if layout == "cell":
        # A cell step packs tokens of several sequences into the shared pool,
        # so the pool bounds its width, not one sequence's capacity.
        max_cells = manifest.get("max_cells")
        if not isinstance(max_cells, int) or max_cells < max_write:
            raise ValueError("off-graph cell layout needs max_cells >= max_write")
    elif max_write > maximum_capacity:
        raise ValueError("off-graph max_write must be in [1, maximum_capacity]")
    layers = manifest.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("off-graph manifest must contain layers")
    by_id = {}
    for layer in layers:
        layer_id = layer.get("layer_id")
        policy = layer.get("policy")
        window = layer.get("window", 0)
        if not isinstance(layer_id, int) or layer_id < 0 or layer_id in by_id:
            raise ValueError("off-graph layer_id must be unique and nonnegative")
        if policy not in ("flat", "ring"):
            raise ValueError(f"invalid off-graph cache policy {policy!r}")
        if policy == "ring" and (not isinstance(window, int) or window <= 0):
            raise ValueError("off-graph ring cache requires a positive window")
        by_id[layer_id] = layer
    manifest["layers_by_id"] = by_id
    return manifest


def ring_attention_mask(
    position: torch.Tensor, buf_size: int, window: int
) -> torch.Tensor:
    """Bool (1, 1, T_q, buf_size) mask: which ring slots each query may read.

    A ring's live region wraps rather than being a prefix, so ``kv_len`` cannot
    *tighten* the sweep once the ring has wrapped; every slot is swept and this
    says which ones count. ``ring_pos`` recovers the logical position a slot
    currently holds -- slots at or before the newest write belong to the current
    lap, later ones to the previous one -- and the rest is ordinary causal plus
    sliding-window on those positions. Sweeping the whole ring is bounded work:
    ``buf_size`` is a constant.

    Note the ``ring_pos >= 0`` term: before the ring wraps it marks every slot
    at or past ``total_written`` dead, which is what lets ``offgraph_step`` pass
    that same bound to sdpa without dropping a live slot.

    Same rule as ``_build_masks`` in the in-graph model
    (examples/models/muse-glimmer/model/model.py); only ``buf_size`` differs,
    since off-graph sizes the ring by ring_physical_capacity() rather than
    twice the window.
    """
    position = position.reshape(-1)
    total_written = position[-1] + 1
    j = torch.arange(buf_size, dtype=position.dtype, device=position.device)
    ring_pos = j + ((total_written - 1 - j) // buf_size) * buf_size
    delta = position.unsqueeze(1) - ring_pos.unsqueeze(0)
    live = (ring_pos >= 0) & (delta >= 0) & (delta < window)
    return live.unsqueeze(0).unsqueeze(0)


def _write(storage: torch.Tensor, slots: torch.Tensor, kv: torch.Tensor) -> None:
    # Index the contiguous BSHD buffer directly. Writing through a BHSD
    # transpose of it makes Inductor copy the whole declared buffer out and
    # back every step, which also reaches past what the runtime allocated.
    storage.index_copy_(1, slots, kv.transpose(1, 2))


def offgraph_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    position: torch.Tensor,
    k_storage: torch.Tensor,
    v_storage: torch.Tensor,
    scale: float,
    buf_size: int,
    mask: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Write this step's K/V into their slots, then attend.

    One step for both policies. A flat layer is a ring that never wraps: its
    ``buf_size`` is the full capacity, so ``position % buf_size`` is the
    position, and it passes no mask, so sdpa applies bottom-right causal from
    ``kv_len`` on device without materializing one. A ring layer passes the
    ``ring_attention_mask`` every ring layer of the step shares.

    ``k``/``v`` are BHSD; the storage is BSHD (see ``LowerOffGraphKVPass``).
    sdpa returns the query's dtype; ``out_dtype`` is the neutral op's requested
    output dtype.
    """
    position = position.reshape(-1)
    slots = position % buf_size
    _write(k_storage, slots, k)
    _write(v_storage, slots, v)
    # A GPU scalar, so the bound still tracks the sequence under CUDA-graph
    # replay while shapes stay static. It is also safe on a ring: before the
    # ring wraps, ring_attention_mask marks every slot at or past
    # ``total_written`` dead via its ``ring_pos >= 0`` guard, and once it wraps
    # sdpa clamps the bound to the buffer, restoring the full sweep. Without it
    # the call misses sdpa's decode split-K dispatch.
    #
    # Positions are contiguous, so this equals position[-1] + 1 at runtime. It
    # is written with the step width from the shape because AOTI autotunes sdpa
    # on generated inputs whose integer data is zero: a bound read entirely
    # from position data tunes as a one-token context and picks a tile that
    # runs long prefill markedly slower.
    kv_len = position[0] + position.shape[0]
    out = torch.ops.triton.sdpa(
        q,
        k_storage.transpose(1, 2),
        v_storage.transpose(1, 2),
        mask,
        0.0,
        mask is None,
        scale,
        True,
        kv_len,
    )
    if out_dtype is not None and out.dtype != out_dtype:
        out = out.to(out_dtype)
    return out


def cell_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    cells: torch.Tensor,
    read_len: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Write this step's K/V into the cells the cache placed it in, then attend.

    Placement and visibility are decided on the host by the runtime cache,
    which writes them into ``cells``, ``mask`` and ``read_len`` before the
    forward: token i lands in cell ``cells[i]`` and attends cell j iff
    ``mask[0, 0, i, j]``. Cells of several sequences interleave in one pool,
    so no causal alignment describes visibility and the mask is always
    explicit. ``read_len`` bounds the sweep to the occupied extent.

    ``cells`` and ``mask`` are declared at the widest step; writes use the
    first K/V-width cells and attention uses the first query-width mask rows.
    ``k``/``v`` are BHSD; the pools are BSHD. sdpa returns
    the query's dtype; ``out_dtype`` is the neutral op's requested output dtype.
    """
    query_width = q.shape[2]
    slots = cells[: k.shape[2]]
    _write(k_pool, slots, k)
    _write(v_pool, slots, v)
    out = torch.ops.triton.sdpa(
        q,
        k_pool.transpose(1, 2),
        v_pool.transpose(1, 2),
        mask[:, :, :query_width, :],
        0.0,
        False,
        scale,
        True,
        read_len,
    )
    if out_dtype is not None and out.dtype != out_dtype:
        out = out.to(out_dtype)
    return out


def batched_sequence_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    position: torch.Tensor,
    k_storage: torch.Tensor,
    v_storage: torch.Tensor,
    seq_cols: torch.Tensor,
    seq_info: torch.Tensor,
    scale: float,
    rows: int,
    columns: int,
    window: int,
    ring: bool,
    causal: bool,
    out_dtype: torch.dtype | None = None,
    context_window: bool = False,
) -> torch.Tensor:
    """Write this step's K/V into each sequence's column, then attend.

    The storage is declared ``[1, rows * columns, H, D]``: row ``r`` of
    column ``c`` is slot ``r * columns + c``, so it is ``[rows, columns, H, D]``
    and grows by whole rows, as the sequence layout's does. Token ``i`` lands
    in column ``seq_cols[i]`` at row ``position % rows`` (a ring layer's rows
    wrap; a flat layer's never do). ``seq_info`` names each segment's query
    rows and length, so every query reads only its own column.

    ``seq_cols`` is declared at the widest step; writes use the first
    K/V-width entries. ``k``/``v`` are BHSD; the storage is BSHD.
    ``context_window`` gives every draft block query the same retained context
    before the block; ordinary attention uses each query's own window.
    """
    position = position.reshape(-1)
    slots = (position % rows) * columns + seq_cols[: k.shape[2]]
    _write(k_storage, slots, k)
    _write(v_storage, slots, v)
    heads, dim = k_storage.shape[2], k_storage.shape[3]
    out = torch.ops.triton.sdpa_varlen(
        q,
        k_storage.view(rows, columns, heads, dim),
        v_storage.view(rows, columns, heads, dim),
        seq_info,
        scale,
        causal,
        window,
        rows if ring else 0,
        context_window=context_window,
    )
    if out_dtype is not None and out.dtype != out_dtype:
        out = out.to(out_dtype)
    return out


def _batched_sequence_step_fn(
    scale: float,
    rows: int,
    columns: int,
    window: int,
    ring: bool,
    causal: bool,
    out_dtype,
    context_window: bool = False,
):
    def fn(q, k, v, position, k_storage, v_storage, seq_cols, seq_info):
        return batched_sequence_step(
            q,
            k,
            v,
            position,
            k_storage,
            v_storage,
            seq_cols,
            seq_info,
            scale,
            rows,
            columns,
            window,
            ring,
            causal,
            out_dtype,
            context_window,
        )

    return fn


def _cell_step_fn(scale: float, out_dtype):
    def fn(q, k, v, k_pool, v_pool, cells, read_len, mask):
        return cell_step(
            q, k, v, k_pool, v_pool, cells, read_len, mask, scale, out_dtype
        )

    return fn


def _mask_fn(buf_size: int, window: int):
    """Freeze the constants in a closure rather than as default arguments.

    ``make_fx`` builds the traced signature from ``co_argcount``, which counts
    defaulted parameters, so a function carrying its constants as defaults is
    judged under-supplied and the trace is rejected before it starts.
    """

    def fn(position):
        return ring_attention_mask(position, buf_size, window)

    return fn


def _step_fn(scale: float, buf_size: int, masked: bool, out_dtype):
    if masked:

        def fn(q, k, v, position, k_storage, v_storage, mask):
            return offgraph_step(
                q,
                k,
                v,
                position,
                k_storage,
                v_storage,
                scale,
                buf_size,
                mask,
                out_dtype,
            )

    else:

        def fn(q, k, v, position, k_storage, v_storage):
            return offgraph_step(
                q,
                k,
                v,
                position,
                k_storage,
                v_storage,
                scale,
                buf_size,
                out_dtype=out_dtype,
            )

    return fn


# Ops that pass a tensor's values through unchanged, which the export of a
# position input may place between the method input and the cache op.
_VALUE_PRESERVING_OPS = (
    "view",
    "reshape",
    "unsqueeze",
    "squeeze",
    "expand",
    "clone",
    "alias",
    "_to_copy",
)


def _source_input(node):
    while node.op == "call_function" and any(
        name in str(node.target) for name in _VALUE_PRESERVING_OPS
    ):
        node = node.args[0]
    return node


def _cell_position_source(node):
    # Edge conversion emits copy variants of the value-preserving view ops.
    # Match overloads exactly rather than admitting arbitrary name substrings.
    preserving = tuple(
        op
        for name in (
            *_VALUE_PRESERVING_OPS,
            "view_copy",
            "unsqueeze_copy",
            "squeeze_copy",
            "expand_copy",
            "alias_copy",
        )
        for op in (
            getattr(torch.ops.aten, name).default,
            getattr(exir_ops.edge.aten, name).default,
        )
    )
    selects = (
        torch.ops.aten.index_select.default,
        exir_ops.edge.aten.index_select.default,
    )
    width = node.meta["val"].shape[0]
    selected = False
    while node.op == "call_function":
        if node.target in preserving:
            node = node.args[0]
        elif node.target in selects and not selected:
            source, dim, index = node.args
            shape = source.meta["val"].shape
            if (
                dim != 0
                or shape[0] != width
                or tuple(node.meta["val"].shape) != tuple(shape)
                or index.meta["val"].dim() != 1
                or index.meta["val"].shape[0] != shape[0]
            ):
                raise ValueError(
                    "off-graph cell position index_select must preserve full width on dim 0"
                )
            selected = True
            node = source
        else:
            break
    return node


class CheckOffGraphKVStepWidthPass:
    """Check the declared step width against the delegated program.

    The runtime reads each step's token count from input ``input`` at ``dim``
    of the delegate. That index is the delegate's own input order, which
    partitioning decides, so it is checked here, after partitioning: every
    cache op's position must come from that input, whose ``dim`` must be the
    position's length.
    """

    requires_exported_program = True

    def __init__(self, value: bytes, layout: str = "sequence") -> None:
        if layout not in ("sequence", *PACKED_LAYOUTS):
            raise ValueError(f"invalid off-graph KV layout {layout!r}")
        self._packed = layout in PACKED_LAYOUTS
        text = value.decode("utf-8")
        fields = text.split(":")
        if len(fields) != 2 or not all(f.isdigit() for f in fields):
            raise ValueError(
                f"{OFFGRAPH_KV_STEP_WIDTH_COMPILE_SPEC} must be "
                f'"input_index:dim", got {text!r}'
            )
        self._input, self._dim = (int(f) for f in fields)

    def __call__(self, exported_program: ExportedProgram) -> ExportedProgram:
        graph = exported_program.graph_module.graph
        user_inputs = set(exported_program.graph_signature.user_inputs)
        inputs = [
            n for n in graph.nodes if n.op == "placeholder" and n.name in user_inputs
        ]
        target = exir_ops.edge.kvcache.update_and_attend.default
        for node in graph.nodes:
            if node.op != "call_function" or node.target != target:
                continue
            if self._input >= len(inputs):
                raise ValueError(
                    f"off-graph KV step width names input {self._input}, but "
                    f"the delegate takes {len(inputs)}"
                )
            declared = inputs[self._input]
            position = node.args[3]
            source = (
                _cell_position_source(position)
                if self._packed
                else _source_input(position)
            )
            if source is not declared:
                raise ValueError(
                    f"off-graph KV step width names input {self._input} "
                    f"({declared.name}), but the cache op's position comes "
                    f"from {source.name}"
                )
            shape = declared.meta["val"].shape
            width = position.meta["val"].shape[0]
            if self._dim >= len(shape) or shape[self._dim] != width:
                raise ValueError(
                    f"off-graph KV step width names dim {self._dim} of "
                    f"{declared.name} {tuple(shape)}, which is not the step's "
                    f"{width} positions"
                )
            k, v = (arg.meta["val"] for arg in node.args[1:3])
            if k.shape[2] != width or v.shape[2] != width:
                raise ValueError(
                    "off-graph KV step width requires matching K/V extent and "
                    f"positions, got {k.shape[2]}, {v.shape[2]} and {width}"
                )
        return exported_program


class LowerOffGraphKVPass:
    requires_exported_program = True

    def __init__(self, manifest: dict[str, Any]) -> None:
        self._manifest = manifest
        self._masks: dict[tuple[str, int, int], Any] = {}
        self._cell_buffers: dict[str, Any] = {}

    @property
    def _cell_layout(self) -> bool:
        return self._manifest.get("layout", "sequence") == "cell"

    @property
    def _batched_sequence_layout(self) -> bool:
        return self._manifest.get("layout", "sequence") == "batched-sequence"

    @property
    def _columns(self) -> int:
        # One per sequence, and a scratch column static-width padding writes.
        return self._manifest["max_seqs"] + 1

    @staticmethod
    def _compile_storage(shape, dtype: torch.dtype, device):
        storage = torch.empty(shape, dtype=dtype, device=device)
        # AOTI needs the logical metadata, while the runtime supplies storage.
        # Nothing can run on it, so lowering must keep AOTI's compile-time
        # autotune (generated inputs) rather than the JIT pass on real constants.
        storage.untyped_storage().resize_(0)
        return storage

    @staticmethod
    def _check_supported(node) -> None:
        """Reject valid neutral-op inputs this lowering does not implement.

        Both would otherwise fail inside the trace with a shape error that says
        nothing about why.
        """
        _, k, v, position = (n.meta["val"] for n in node.args[0:4])
        if v.shape[-1] != k.shape[-1]:
            raise ValueError(
                "off-graph KV lowering requires v_head_dim == head_dim, got "
                f"{v.shape[-1]} and {k.shape[-1]}"
            )
        if position.dim() not in (1, 2) or (
            position.dim() == 2 and position.shape[1] != 1
        ):
            raise ValueError(
                "off-graph KV lowering supports one position component per "
                f"token ([T] or [T, 1]), got {tuple(position.shape)}"
            )

    def _layer_capacity(self, layer: dict[str, Any]) -> int:
        if self._cell_layout:
            # A window bounds what a query sees, not where its token lives: a
            # cell layout keeps every layer's history in the one pool.
            return self._manifest["max_cells"]
        if layer["policy"] == "ring":
            rows = ring_physical_capacity(layer["window"], self._manifest["max_write"])
            if self._batched_sequence_layout:
                # A sequence never reaches past its capacity, so a ring that
                # large never wraps.
                rows = min(rows, self._manifest["maximum_capacity"])
            return rows
        return self._manifest["maximum_capacity"]

    def _storage_nodes(
        self, exported_program: ExportedProgram, node, layer: dict[str, Any]
    ):
        layer_id = layer["layer_id"]
        # Shape and dtype come from the step's own K, so the storage cannot
        # disagree with what gets written into it. The manifest only carries
        # what the tensors cannot say: how the layer retains history, and how
        # far it may grow.
        kv = node.args[1].meta["val"]  # BHSD
        if kv.dtype != torch.bfloat16:
            raise ValueError(
                f"off-graph KV cache currently requires bfloat16, got {kv.dtype}"
            )
        capacity = self._layer_capacity(layer)
        # Declared at the maximum capacity, sequence-major (BSHD). Every access
        # is bounded by kv_len along the sequence dim, and with the sequence
        # outermost no other stride depends on capacity, so the runtime may
        # back this with an allocation holding only the rows written so far
        # and grow it. A BHSD declaration would put head h at h * capacity,
        # past any smaller allocation.
        if self._batched_sequence_layout:
            # Rows outermost, so the runtime grows every column at once.
            capacity *= self._columns
        shape = (kv.shape[0], capacity, kv.shape[1], kv.shape[3])
        prefix = f"{OFFGRAPH_KV_FQN_PREFIX}layer_{layer_id}"
        names = (f"{prefix}_k", f"{prefix}_v")
        values = (
            self._compile_storage(shape, kv.dtype, kv.device),
            self._compile_storage(shape, kv.dtype, kv.device),
        )
        return [
            self._runtime_buffer(exported_program, name, value)
            for name, value in zip(names, values)
        ]

    @staticmethod
    def _runtime_buffer(exported_program: ExportedProgram, name: str, value):
        graph = exported_program.graph_module.graph
        with graph.inserting_before(next(iter(graph.nodes))):
            return create_constant_placeholder(
                exp_program=exported_program,
                graph=graph,
                name=name,
                kind=InputKind.BUFFER,
                data=value,
                persistent_buffer=False,
            )

    def _cell_buffer(self, exported_program, name: str, shape, dtype, device):
        """A step buffer of a packed layout, declared once per program.

        Unlike the pools these are small, so they carry real (zero) bytes at
        compile time: Inductor reads small constants while it compiles, and
        they stay runtime-owned regardless, since the weight collector never
        serializes a runtime-owned FQN.
        """
        node = self._cell_buffers.get(name)
        if node is None:
            node = self._runtime_buffer(
                exported_program, name, torch.zeros(shape, dtype=dtype, device=device)
            )
            self._cell_buffers[name] = node
        return node

    def _lower_cell(
        self, exported_program, node, layer, k_storage, v_storage, out_dtype
    ):
        max_write = self._manifest["max_write"]
        max_cells = self._manifest["max_cells"]
        device = node.args[1].meta["val"].device
        window = layer["window"] if layer["policy"] == "ring" else 0
        cells = self._cell_buffer(
            exported_program, OFFGRAPH_KV_CELLS_FQN, (max_write,), torch.int64, device
        )
        read_len = self._cell_buffer(
            exported_program, OFFGRAPH_KV_READ_LEN_FQN, (1,), torch.int64, device
        )
        mask = self._cell_buffer(
            exported_program,
            offgraph_kv_mask_fqn(window),
            (1, 1, max_write, max_cells),
            torch.bool,
            device,
        )
        call_args = (
            *node.args[0:3],  # q, k, v
            k_storage,
            v_storage,
            cells,
            read_len,
            mask,
        )
        example_args = tuple(n.meta["val"] for n in call_args)
        return _cell_step_fn(node.args[5], out_dtype), call_args, example_args

    def _lower_batched_sequence(
        self, exported_program, node, layer, k_storage, v_storage, out_dtype
    ):
        device = node.args[1].meta["val"].device
        seq_cols = self._cell_buffer(
            exported_program,
            OFFGRAPH_KV_SEQ_COLS_FQN,
            (self._manifest["max_write"],),
            torch.int64,
            device,
        )
        seq_info = self._cell_buffer(
            exported_program,
            OFFGRAPH_KV_SEQ_INFO_FQN,
            (self._manifest["max_segments"], 4),
            torch.int32,
            device,
        )
        call_args = (*node.args[0:4], k_storage, v_storage, seq_cols, seq_info)
        example_args = tuple(n.meta["val"] for n in call_args)
        ring = layer["policy"] == "ring"
        fn = _batched_sequence_step_fn(
            node.args[5],
            self._layer_capacity(layer),
            self._columns,
            layer["window"] if ring else 0,
            ring,
            self._manifest["causal"],
            out_dtype,
            self._manifest.get("context_window", False),
        )
        return fn, call_args, example_args

    def _ring_mask(self, graph, position, buf_size: int, window: int, before):
        """Emit the ring mask once and share it across layers that match.

        It depends only on ``position``, so it can be hoisted above the cache
        writes and reused; every ring layer of a given window wants the same
        tensor, and at prefill widths that tensor is large enough that building
        one per layer dominates the step.
        """
        key = (position.name, buf_size, window)
        cached = self._masks.get(key)
        if cached is not None:
            return cached
        mask = self._inline(
            graph,
            _mask_fn(buf_size, window),
            (position.meta["val"],),
            (position,),
            before,
        )
        self._masks[key] = mask
        return mask

    def _lower_sequence(self, graph, node, layer, k_storage, v_storage, out_dtype):
        inputs = list(node.args[0:4])  # q, k, v, position
        call_args = (*inputs, k_storage, v_storage)
        example_args = tuple(n.meta["val"] for n in call_args)
        buf_size = self._layer_capacity(layer)
        masked = layer["policy"] == "ring"
        if masked:
            mask = self._ring_mask(graph, inputs[3], buf_size, layer["window"], node)
            call_args = (*call_args, mask)
            example_args = (*example_args, mask.meta["val"])
        return (
            _step_fn(node.args[5], buf_size, masked, out_dtype),
            call_args,
            example_args,
        )

    @staticmethod
    def _inline(graph, fn, example_args, call_args, before):
        """Trace ``fn`` and splice its body in ahead of ``before``.

        The decomposition is written once, as ordinary tensor code, and the
        tests import the very same functions -- so a test cannot agree with a
        mask the pass does not actually emit.
        """
        traced = make_fx(fn)(*example_args)
        env = dict(
            zip(
                [n for n in traced.graph.nodes if n.op == "placeholder"],
                call_args,
            )
        )
        result = None
        with graph.inserting_before(before):
            for traced_node in traced.graph.nodes:
                if traced_node.op == "placeholder":
                    continue
                if traced_node.op == "output":
                    result = torch.fx.map_arg(traced_node.args[0], env.get)
                    continue
                env[traced_node] = graph.node_copy(traced_node, lambda n: env[n])
        return result

    def __call__(self, exported_program: ExportedProgram) -> ExportedProgram:
        graph_module = exported_program.graph_module
        # Mask and buffer nodes belong to one graph; never carry them into
        # another.
        self._masks = {}
        self._cell_buffers = {}
        target = exir_ops.edge.kvcache.update_and_attend.default
        modified = False
        for node in list(graph_module.graph.nodes):
            if node.op != "call_function" or node.target != target:
                continue
            layer_id = node.args[4]
            if not isinstance(layer_id, int):
                raise ValueError("off-graph layer_id must be a graph constant")
            layer = self._manifest["layers_by_id"].get(layer_id)
            if layer is None:
                raise ValueError(f"off-graph manifest has no layer {layer_id}")

            self._check_supported(node)
            out_dtype = (
                node.args[6] if len(node.args) > 6 else node.kwargs.get("out_dtype")
            )
            k_storage, v_storage = self._storage_nodes(exported_program, node, layer)
            if self._cell_layout:
                fn, call_args, example_args = self._lower_cell(
                    exported_program, node, layer, k_storage, v_storage, out_dtype
                )
            elif self._batched_sequence_layout:
                fn, call_args, example_args = self._lower_batched_sequence(
                    exported_program, node, layer, k_storage, v_storage, out_dtype
                )
            else:
                fn, call_args, example_args = self._lower_sequence(
                    graph_module.graph, node, layer, k_storage, v_storage, out_dtype
                )

            new_node = self._inline(
                graph_module.graph, fn, example_args, call_args, node
            )
            new_node.meta = node.meta.copy()
            node.replace_all_uses_with(new_node)
            graph_module.graph.erase_node(node)
            modified = True
        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
        return exported_program
