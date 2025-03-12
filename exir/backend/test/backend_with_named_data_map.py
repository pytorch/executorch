# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, final, List, Tuple

import torch
from executorch.exir._serialize._named_data_store import NamedDataStore

from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.canonical_partitioners.pattern_op_partitioner import (
    generate_pattern_op_partitions,
)

from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.graph_module import get_control_flow_submodules
from torch.export.exported_program import ExportedProgram
from torch.fx.passes.operator_support import OperatorSupportBase


# Backend details are final (cannot be subclassed).
@final
class BackendWithNamedDataMap(BackendDetails):
    """
    Test Backend for Named Data Map Functionality

    This backend returns no processed_bytes, instead it uses
    the named data store and serializes the name of the op
    as the key and the data as its code value
    """

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        op_codes = {
            exir_ops.edge.aten.sin.default: 0,
            exir_ops.edge.aten.add.Tensor: 1,
            exir_ops.edge.aten.sub.Tensor: 2,
            exir_ops.edge.aten.mul.Tensor: 3,
            exir_ops.edge.aten.div.Tensor: 4,
        }
        ndm = NamedDataStore()
        for node in edge_program.graph.nodes:
            if node.op == "call_function":
                if node.target in op_codes.keys():
                    ndm.add_named_data(
                        node.target.__name__, bytes(op_codes[node.target])
                    )

        return PreprocessResult(
            processed_bytes=bytes(b""),
            debug_handle_map={},
            data_store_output=ndm.get_named_data_store_output(),
        )


class SimpleOperatorSupport(OperatorSupportBase):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        return node.op == "call_function" and node.target in [
            exir_ops.edge.aten.sin.default,
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.sub.Tensor,
            exir_ops.edge.aten.mul.Tensor,
            exir_ops.edge.aten.div.Tensor,
        ]


@final
class BackendWithNDMPartitioner(Partitioner):
    def __init__(self) -> None:
        self._op_support = SimpleOperatorSupport()
        self.backend_id = BackendWithNamedDataMap.__name__

    def _partition_gm(
        self, graph_module: torch.fx.GraphModule, id_start: int = 0
    ) -> Tuple[int, Dict[str, DelegationSpec]]:
        partition_tags: Dict[str, DelegationSpec] = {}
        partition_list = generate_pattern_op_partitions(
            graph_module, op_support=self._op_support
        )

        num_partitions_in_gm = len(partition_list)
        for partition in partition_list:
            curr_par_id = partition.id or 0
            delegation_tag = f"tag_{curr_par_id + id_start}"
            for node in partition.nodes:
                node.meta["delegation_tag"] = delegation_tag
            delegation_spec = DelegationSpec(self.backend_id, [])
            partition_tags[delegation_tag] = delegation_spec

        start_idx_for_submodules = num_partitions_in_gm
        for _, submodule, _ in get_control_flow_submodules(graph_module):
            start_idx_for_submodules, ret_partition_tags = self._partition_gm(
                submodule, start_idx_for_submodules
            )
            partition_tags.update(ret_partition_tags)

        return start_idx_for_submodules, partition_tags

    def partition(self, edge_program: ExportedProgram) -> PartitionResult:
        _, partition_tags = self._partition_gm(edge_program.graph_module)
        return PartitionResult(
            tagged_exported_program=edge_program,
            partition_tags=partition_tags,
        )
