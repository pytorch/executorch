# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from typing import Any

import torch
from executorch.backends.transforms.utils import create_constant_placeholder
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind

from executorch.backends.cuda.triton.kernels.offgraph_kv import (
    FLAT_CACHE,
    RING_CACHE,
)


OFFGRAPH_KV_COMPILE_SPEC = "offgraph_kv_manifest"
OFFGRAPH_KV_FQN_PREFIX = "__et_offgraph_kv_"


def parse_offgraph_kv_manifest(value: bytes) -> dict[str, Any]:
    manifest = json.loads(value.decode("utf-8"))
    if manifest.get("version") != 1:
        raise ValueError("off-graph KV manifest version must be 1")
    if manifest.get("dtype") != "bfloat16":
        raise ValueError("off-graph KV cache currently requires bfloat16")
    maximum_capacity = manifest.get("maximum_capacity")
    initial_capacity = manifest.get("initial_capacity")
    if not isinstance(maximum_capacity, int) or maximum_capacity <= 0:
        raise ValueError("off-graph maximum_capacity must be positive")
    if (
        not isinstance(initial_capacity, int)
        or initial_capacity <= 0
        or initial_capacity > maximum_capacity
    ):
        raise ValueError("off-graph initial_capacity is invalid")
    layers = manifest.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("off-graph manifest must contain layers")
    by_id = {}
    for layer in layers:
        layer_id = layer.get("layer_id")
        policy = layer.get("policy")
        heads = layer.get("num_kv_heads")
        head_dim = layer.get("head_dim")
        window = layer.get("window", 0)
        if not isinstance(layer_id, int) or layer_id < 0 or layer_id in by_id:
            raise ValueError("off-graph layer_id must be unique and nonnegative")
        if policy not in ("flat", "ring"):
            raise ValueError(f"invalid off-graph cache policy {policy!r}")
        if not isinstance(heads, int) or heads <= 0:
            raise ValueError("off-graph num_kv_heads must be positive")
        if not isinstance(head_dim, int) or head_dim <= 0:
            raise ValueError("off-graph head_dim must be positive")
        if policy == "ring" and (not isinstance(window, int) or window <= 0):
            raise ValueError("off-graph ring cache requires a positive window")
        by_id[layer_id] = layer
    manifest["layers_by_id"] = by_id
    return manifest


class LowerOffGraphKVPass:
    requires_exported_program = True

    def __init__(self, manifest: dict[str, Any]) -> None:
        self._manifest = manifest

    @staticmethod
    def _compile_storage(numel: int, dtype: torch.dtype, device: torch.device):
        storage = torch.empty(numel, dtype=dtype, device=device)
        # AOTI needs the logical metadata, while the runtime supplies storage.
        storage.untyped_storage().resize_(0)
        return storage

    def _storage_nodes(
        self, exported_program: ExportedProgram, node, layer: dict[str, Any]
    ):
        graph = exported_program.graph_module.graph
        layer_id = layer["layer_id"]
        heads = layer["num_kv_heads"]
        head_dim = layer["head_dim"]
        if layer["policy"] == "ring":
            capacity = layer["window"] * 2
        else:
            capacity = self._manifest["maximum_capacity"]
        device = node.args[0].meta["val"].device
        numel = heads * capacity * head_dim
        prefix = f"{OFFGRAPH_KV_FQN_PREFIX}layer_{layer_id}"
        names = (f"{prefix}_k", f"{prefix}_v", f"{prefix}_capacity")
        values = (
            self._compile_storage(numel, torch.bfloat16, device),
            self._compile_storage(numel, torch.bfloat16, device),
            torch.tensor([capacity], dtype=torch.int64, device=device),
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

    def __call__(self, exported_program: ExportedProgram) -> ExportedProgram:
        graph_module = exported_program.graph_module
        target = exir_ops.edge.kvcache.update_and_attend.default
        replacement = torch.ops.triton.cuda_offgraph_update_and_attend.default
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
            with graph_module.graph.inserting_before(node):
                k_storage, v_storage, capacity = self._storage_nodes(
                    exported_program, node, layer
                )
                policy = RING_CACHE if layer["policy"] == "ring" else FLAT_CACHE
                new_node = graph_module.graph.call_function(
                    replacement,
                    args=(
                        node.args[0],
                        node.args[1],
                        node.args[2],
                        node.args[3],
                        k_storage,
                        v_storage,
                        capacity,
                        policy,
                        layer.get("window", 0),
                        node.args[5],
                        node.args[6],
                    ),
                )
                new_node.meta = node.meta.copy()
            node.replace_all_uses_with(new_node)
            graph_module.graph.erase_node(node)
            modified = True
        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
        return exported_program
