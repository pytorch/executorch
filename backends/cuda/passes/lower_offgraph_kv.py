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


def parse_offgraph_kv_manifest(value: bytes) -> dict[str, Any]:
    manifest = json.loads(value.decode("utf-8"))
    if manifest.get("version") != 1:
        raise ValueError("off-graph KV manifest version must be 1")
    maximum_capacity = manifest.get("maximum_capacity")
    if not isinstance(maximum_capacity, int) or maximum_capacity <= 0:
        raise ValueError("off-graph maximum_capacity must be positive")
    max_write = manifest.get("max_write")
    if not isinstance(max_write, int) or not 0 < max_write <= maximum_capacity:
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
) -> torch.Tensor:
    """Write this step's K/V into their slots, then attend.

    One step for both policies. A flat layer is a ring that never wraps: its
    ``buf_size`` is the full capacity, so ``position % buf_size`` is the
    position, and it passes no mask, so sdpa applies bottom-right causal from
    ``kv_len`` on device without materializing one. A ring layer passes the
    ``ring_attention_mask`` every ring layer of the step shares.

    ``k``/``v`` are BHSD; the storage is BSHD (see ``LowerOffGraphKVPass``).
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
    kv_len = position[-1] + 1
    return torch.ops.triton.sdpa(
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


def _mask_fn(buf_size: int, window: int):
    """Freeze the constants in a closure rather than as default arguments.

    ``make_fx`` builds the traced signature from ``co_argcount``, which counts
    defaulted parameters, so a function carrying its constants as defaults is
    judged under-supplied and the trace is rejected before it starts.
    """

    def fn(position):
        return ring_attention_mask(position, buf_size, window)

    return fn


def _step_fn(scale: float, buf_size: int, masked: bool):
    if masked:

        def fn(q, k, v, position, k_storage, v_storage, mask):
            return offgraph_step(
                q, k, v, position, k_storage, v_storage, scale, buf_size, mask
            )

    else:

        def fn(q, k, v, position, k_storage, v_storage):
            return offgraph_step(
                q, k, v, position, k_storage, v_storage, scale, buf_size
            )

    return fn


class LowerOffGraphKVPass:
    requires_exported_program = True

    def __init__(self, manifest: dict[str, Any]) -> None:
        self._manifest = manifest
        self._masks: dict[tuple[str, int, int], Any] = {}

    @staticmethod
    def _compile_storage(shape, dtype: torch.dtype, device):
        storage = torch.empty(shape, dtype=dtype, device=device)
        # AOTI needs the logical metadata, while the runtime supplies storage.
        storage.untyped_storage().resize_(0)
        return storage

    def _layer_capacity(self, layer: dict[str, Any]) -> int:
        if layer["policy"] == "ring":
            return ring_physical_capacity(layer["window"], self._manifest["max_write"])
        return self._manifest["maximum_capacity"]

    def _storage_nodes(
        self, exported_program: ExportedProgram, node, layer: dict[str, Any]
    ):
        graph = exported_program.graph_module.graph
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
        shape = (kv.shape[0], capacity, kv.shape[1], kv.shape[3])
        prefix = f"{OFFGRAPH_KV_FQN_PREFIX}layer_{layer_id}"
        names = (f"{prefix}_k", f"{prefix}_v")
        values = (
            self._compile_storage(shape, kv.dtype, kv.device),
            self._compile_storage(shape, kv.dtype, kv.device),
        )
        result = []
        first_node = next(iter(graph.nodes))
        for name, value in zip(names, values):
            with graph.inserting_before(first_node):
                result.append(
                    create_constant_placeholder(
                        exp_program=exported_program,
                        graph=graph,
                        name=name,
                        kind=InputKind.BUFFER,
                        data=value,
                        persistent_buffer=False,
                    )
                )
        return result

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
        # Mask nodes belong to one graph; never carry them into another.
        self._masks = {}
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

            inputs = list(node.args[0:4])  # q, k, v, position
            scale = node.args[5]
            k_storage, v_storage = self._storage_nodes(exported_program, node, layer)
            call_args = (*inputs, k_storage, v_storage)
            example_args = tuple(n.meta["val"] for n in call_args)

            buf_size = self._layer_capacity(layer)
            masked = layer["policy"] == "ring"
            if masked:
                mask = self._ring_mask(
                    graph_module.graph, inputs[3], buf_size, layer["window"], node
                )
                call_args = (*call_args, mask)
                example_args = (*example_args, mask.meta["val"])
            fn = _step_fn(scale, buf_size, masked)

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
