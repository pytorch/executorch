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
    at or past ``total_written`` dead, which is what lets ``ring_step`` pass
    that same bound to sdpa without dropping a live slot.

    Same rule as ``_build_masks`` in the in-graph model
    (examples/models/muse-glimmer/model/model.py); only ``buf_size`` differs,
    since off-graph sizes the ring by ring_physical_capacity() rather than
    twice the window.
    """
    total_written = position[-1] + 1
    j = torch.arange(buf_size, dtype=position.dtype, device=position.device)
    ring_pos = j + ((total_written - 1 - j) // buf_size) * buf_size
    delta = position.unsqueeze(1) - ring_pos.unsqueeze(0)
    live = (ring_pos >= 0) & (delta >= 0) & (delta < window)
    return live.unsqueeze(0).unsqueeze(0)


def flat_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    position: torch.Tensor,
    k_storage: torch.Tensor,
    v_storage: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Append this step's K/V at their logical positions, then attend."""
    k_storage.index_copy_(2, position, k)
    v_storage.index_copy_(2, position, v)
    # A GPU scalar, so the bound still tracks the sequence under CUDA-graph
    # replay; the buffer's shape stays static.
    kv_len = position[-1] + 1
    return torch.ops.triton.sdpa(
        q, k_storage, v_storage, None, 0.0, True, scale, True, kv_len
    )


def ring_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    position: torch.Tensor,
    k_storage: torch.Tensor,
    v_storage: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
    buf_size: int,
) -> torch.Tensor:
    """Write into wrapped slots, then attend over the ring behind ``mask``.

    The mask is passed in rather than built here because every ring layer in a
    step wants the same one: it depends only on ``position``, so building it
    per layer would materialize an identical (1,1,T_q,buf_size) tensor once per
    layer. The in-graph model shares one the same way.
    """
    k_storage.index_copy_(2, position % buf_size, k)
    v_storage.index_copy_(2, position % buf_size, v)
    # Same bound flat_step passes, and safe on a ring for the reason the mask
    # already relies on: before the ring wraps, ``position % buf_size`` is the
    # identity, and ring_attention_mask marks every slot at or past
    # ``total_written`` dead via its ``ring_pos >= 0`` guard -- so bounding the
    # sweep here cannot drop a live slot. Once wrapped, total_written exceeds
    # the buffer and sdpa clamps the bound to it, restoring the full sweep.
    # Without it the call misses the decode split-K dispatch and every ring
    # layer falls onto the general, prefill-shaped kernel.
    kv_len = position[-1] + 1
    return torch.ops.triton.sdpa(
        q, k_storage, v_storage, mask, 0.0, False, scale, True, kv_len
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


def _ring_fn(scale: float, buf_size: int):
    def fn(q, k, v, position, k_storage, v_storage, mask):
        return ring_step(q, k, v, position, k_storage, v_storage, mask, scale, buf_size)

    return fn


def _flat_fn(scale: float):
    def fn(q, k, v, position, k_storage, v_storage):
        return flat_step(q, k, v, position, k_storage, v_storage, scale)

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
        # A declared 4-D shape rather than a flat blob: index_copy_ and sdpa
        # both address through these strides, so the runtime's allocation has
        # to match the declaration or both will run off the end of it.
        shape = (kv.shape[0], kv.shape[1], capacity, kv.shape[3])
        prefix = f"{OFFGRAPH_KV_FQN_PREFIX}layer_{layer_id}"
        names = (f"{prefix}_k", f"{prefix}_v", f"{prefix}_capacity")
        values = (
            self._compile_storage(shape, kv.dtype, kv.device),
            self._compile_storage(shape, kv.dtype, kv.device),
            torch.tensor([capacity], dtype=torch.int64, device=kv.device),
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
            k_storage, v_storage, _capacity = self._storage_nodes(
                exported_program, node, layer
            )
            call_args = (*inputs, k_storage, v_storage)
            example_args = tuple(n.meta["val"] for n in call_args)

            if layer["policy"] == "ring":
                buf_size = self._layer_capacity(layer)
                window = layer["window"]
                mask = self._ring_mask(
                    graph_module.graph, inputs[3], buf_size, window, node
                )
                call_args = (*call_args, mask)
                example_args = (*example_args, mask.meta["val"])
                fn = _ring_fn(scale, buf_size)

            else:
                fn = _flat_fn(scale)

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
