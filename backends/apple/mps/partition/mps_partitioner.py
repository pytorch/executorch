#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

import logging
from typing import Any, cast, Dict, List, Union

import torch
from executorch.backends.apple.mps import MPSBackend
from executorch.backends.apple.mps.operators.node_visitor import get_node_visitors
from executorch.backends.transforms import get_shape
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.backend.canonical_partitioners.pattern_op_partitioner import (
    generate_partitions_from_list_of_nodes,
)
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export.exported_program import ExportedProgram
from torch.fx.passes.infra.partitioner import Partition
from torch.fx.passes.operator_support import OperatorSupportBase

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

# ops implemented as Metal kernels.
METAL_KERNELS = [
    exir_ops.edge.aten.index.Tensor,
    exir_ops.edge.aten.index_put.default,
]


class MPSOperatorSupport(OperatorSupportBase):
    def __init__(self, edge_program: torch.export.ExportedProgram, compiler_specs):
        self.node_visitors = get_node_visitors(edge_program)
        self.edge_program = edge_program

    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        if node.op != "call_function":
            return False

        if node.target.__name__ not in self.node_visitors:
            logging.debug(f"[UNSUPPORTED] Node {node.target.__name__} not supported")
            return False

        return True


class MPSPartitioner(Partitioner):
    def __init__(self, compile_specs: List[CompileSpec]) -> None:
        self.compile_specs = compile_specs
        self.delegation_spec = DelegationSpec(MPSBackend.__name__, compile_specs)
        self.partition_tags: Dict[str, DelegationSpec] = {}

    def generate_partitions(self, edge_program: ExportedProgram) -> List[Any]:
        self.supported_ops = MPSOperatorSupport(
            edge_program=edge_program, compiler_specs=self.delegation_spec.compile_specs
        )
        return generate_partitions_from_list_of_nodes(
            edge_program.graph_module,
            op_support=self.supported_ops,
        )

    def mps_graph_advanced_indexing_support(self, node: torch.fx.Node):
        num_indices = 0
        tensors = cast(List[torch.fx.Node], node.args[1])
        input = cast(torch.fx.Node, node.args[0])
        for t in tensors:
            if t is not None:
                num_indices += 1
        # Can dispatch to MPSGraph if the length of the slices is equal
        # to the number of dimensions of the sliced tensors, or only one
        # slice is present. All other cases will fallback to a Metal kernel.
        if num_indices == len(get_shape(input)) or num_indices == 1:
            return True

        return False

    def use_metal_kernel(self, node: torch.fx.Node):
        if node.target in METAL_KERNELS:
            if (
                node.target == exir_ops.edge.aten.index.Tensor
                or node.target == exir_ops.edge.aten.index_put.default
            ):
                if not self.mps_graph_advanced_indexing_support(node):
                    return True
        return False

    def tag_nodes(self, partitions: List[Partition]) -> None:
        for partition in partitions:
            crt_partition_counter = 0
            for node in partition.nodes:
                delegation_tag = f"mps_{partition.id}"
                if self.use_metal_kernel(node):
                    logging.warning(f"[WARNING] Using Metal kernel for op {node.name}!")
                    # Partition the Metal kernel into a separate partition
                    crt_partition_counter += 1
                    delegation_tag = (
                        f"{delegation_tag}_metal_kernel_{crt_partition_counter}"
                    )
                    crt_partition_counter += 1
                else:
                    delegation_tag = f"{delegation_tag}_{crt_partition_counter}"

                node.meta["delegation_tag"] = delegation_tag
                self.partition_tags[delegation_tag] = self.delegation_spec

    @staticmethod
    def check_partitions(partitions: Union[dict, list]) -> bool:
        pl = len(partitions)
        if pl == 0:
            logging.warning("Nothing can be partitioned!")
        else:
            logging.info(f"Found {pl} subgraphs to be partitioned.")
        return pl != 0

    # override
    def partition(self, edge_program: ExportedProgram) -> PartitionResult:
        partitions = self.generate_partitions(edge_program=edge_program)
        if self.check_partitions(partitions):
            self.tag_nodes(partitions)
            # Tag constant data that are used by the supported ops in MPS backend.
            tag_constant_data(edge_program)
        x = PartitionResult(
            tagged_exported_program=edge_program, partition_tags=self.partition_tags
        )

        return x
