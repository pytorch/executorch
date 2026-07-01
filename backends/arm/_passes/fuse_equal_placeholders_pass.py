# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
from collections import defaultdict
from typing import Set, Type

import torch

from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import (
    get_constant_placeholder_kind,
    get_param_tensor,
    is_param_node,
)
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.backends.transforms.utils import (
    create_constant_placeholder,
    delete_constant_placeholder,
)
from executorch.exir import ExportedProgram
from executorch.exir.pass_base import ExportPass, PassResult


class FuseEqualPlaceholdersPass(ArmPass):
    """This pass optimizes memory usage by finding constant placeholders
    pointing to identical tensors and fusing them to one single placeholder with
    multiple users, using a cache for faster comparison.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def __init__(self, exported_program: ExportedProgram, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exported_program = exported_program

    def _get_param_key(
        self, node: torch.fx.Node, include_data_hash: bool
    ) -> tuple[tuple[object, ...], torch.Tensor] | None:
        if not is_param_node(self.exported_program, node):
            return None

        tensor = get_param_tensor(self.exported_program, node)
        if tensor is None:
            return None

        key: tuple[object, ...] = (
            node.meta.get(TosaSpecialDtype.meta_key(), None),
            str(tensor.dtype),
            tuple(tensor.shape),
        )
        if include_data_hash:
            t_cpu = tensor.cpu().contiguous().flatten().view(dtype=torch.uint8)
            data_hash = hashlib.sha1(
                t_cpu.numpy().tobytes(), usedforsecurity=False
            ).hexdigest()
            key = (*key, data_hash)
        return key, tensor

    def should_run_pass(self, graph_module: torch.fx.GraphModule) -> bool:
        seen: set[tuple[object, ...]] = set()
        for node in graph_module.graph.nodes:
            param_key = self._get_param_key(node, include_data_hash=False)
            if param_key is None:
                continue
            key, _ = param_key
            if key in seen:
                # Metadata matches are only a cheap pre-scan; call() hashes the
                # tensor bytes before actually fusing placeholders.
                return True
            seen.add(key)
        return False

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False

        # Build a cache of params: mapping hash_key -> list of (node, tensor)
        hash_buckets = defaultdict(list)
        for node in graph_module.graph.nodes:
            param_key = self._get_param_key(node, include_data_hash=True)
            if param_key is None:
                continue
            key, tensor = param_key
            hash_buckets[key].append((node, tensor))

        # For each bucket with more than one entry, fuse:
        for nodes_tensors in hash_buckets.values():
            if len(nodes_tensors) < 2:
                continue

            # Create a new placeholder from first in list of equal placeholders.
            rep_node, rep_tensor = nodes_tensors[0]
            common_name = rep_node.name + "_common"
            common_kind = get_constant_placeholder_kind(self.exported_program, rep_node)
            common_persistent = True
            with graph_module.graph.inserting_before(rep_node):
                common_node = create_constant_placeholder(
                    self.exported_program,
                    graph_module.graph,
                    common_name,
                    common_kind,
                    rep_tensor,
                    common_persistent,
                )

                # TBD: Find a principled way to merge node.meta across all fused node
                # For now, i specifically transfer over the TosaSpecialDtype.meta_key() of the rep_node
                if TosaSpecialDtype.meta_key() in rep_node.meta:
                    common_node.meta[TosaSpecialDtype.meta_key()] = rep_node.meta[
                        TosaSpecialDtype.meta_key()
                    ]

            # Replace uses and delete duplicates
            for node, _ in nodes_tensors:
                node.replace_all_uses_with(common_node)
                delete_constant_placeholder(self.exported_program, node)
                modified = True

        if modified:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module=graph_module, modified=modified)
