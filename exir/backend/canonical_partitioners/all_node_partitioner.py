# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import torch
from executorch.exir.backend.backend_details import ExportedProgram
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from torch._export.utils import is_buffer, is_lifted_tensor_constant, is_param


def is_non_tensor_placeholder(node: torch.fx.Node, ep: ExportedProgram) -> bool:
    """
    Returns true if the node is a placeholder node and it is not a tensor
    """
    return node.op == "placeholder" and not (
        is_param(ep, node) or is_buffer(ep, node) or is_lifted_tensor_constant(ep, node)
    )


class AllNodePartitioner(Partitioner):
    def __init__(
        self,
        backend_id: str,
        compile_specs: List[CompileSpec],
    ):
        """
        Partitioner that lowers every single node in the graph module to the
        specified backend_id
        """
        super().__init__()
        self.delegation_spec = DelegationSpec(backend_id, compile_specs)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        # tag all nodes
        partition_tags: Dict[str, DelegationSpec] = {}
        for node in exported_program.graph_module.graph.nodes:
            if is_non_tensor_placeholder(node, exported_program) or node.op == "output":
                continue

            delegation_tag = self.delegation_spec.backend_id
            node.meta["delegation_tag"] = delegation_tag
            partition_tags[delegation_tag] = self.delegation_spec

        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )
