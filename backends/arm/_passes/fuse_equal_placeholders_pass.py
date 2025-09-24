# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
from collections import defaultdict
from typing import Set, Type

import torch

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


class FuseEqualPlaceholdersPass(ExportPass):
    """
    This pass optimizes memory usage by finding constant placeholders
    pointing to identical tensors and fusing them to one single placeholder
    with multiple users, using a cache for faster comparison.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def __init__(self, exported_program: ExportedProgram):
        self.exported_program = exported_program
        super().__init__()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False

        # Build a cache of params: mapping hash_key -> list of (node, tensor)
        hash_buckets = defaultdict(list)
        for node in graph_module.graph.nodes:
            if not is_param_node(self.exported_program, node):
                continue
            tensor = get_param_tensor(self.exported_program, node)
            if tensor is None:
                continue
            # Create a lightweight fingerprint: dtype + shape + SHA1 of raw bytes
            # Ensure tensor is on CPU and contiguous

            # ensure we don't merge any special case int48_t tensors with int32_t tensors
            # since int48_t tensors needs to be instantiated separately.
            is_int48 = node.meta.get(TosaSpecialDtype.meta_key(), None)
            t_cpu = tensor.detach().cpu().contiguous()
            data_bytes = t_cpu.numpy().tobytes()
            key = (
                is_int48,
                str(t_cpu.dtype),
                tuple(t_cpu.shape),
                hashlib.sha1(data_bytes).hexdigest(),
            )
            hash_buckets[key].append((node, t_cpu))

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

            # Replace uses and delete duplicates
            for node, _ in nodes_tensors:
                node.replace_all_uses_with(common_node)
                delete_constant_placeholder(self.exported_program, node)
                modified = True

        if modified:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module=graph_module, modified=modified)
