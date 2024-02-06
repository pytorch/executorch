import logging
import operator
import os
from typing import final

import torch
from executorch.backends.arm.arm_backend import ArmBackend
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export.exported_program import ExportedProgram
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner

from torch.fx.passes.operator_support import OperatorSupportBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
TOSA_DBG_VERBOSE = os.environ.get("TOSA_DBG_VERBOSE") == "1"
if TOSA_DBG_VERBOSE:
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)


class TOSASupportedOperators(OperatorSupportBase):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        supported = node.op == "call_function" and node.target in [
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.addmm.default,
            exir_ops.edge.aten.permute_copy.default,
            exir_ops.edge.aten.hardtanh.default,
            exir_ops.edge.aten.convolution.default,
            exir_ops.edge.aten.div.Tensor,
            exir_ops.edge.aten._native_batch_norm_legit_no_training.default,
            exir_ops.edge.aten.avg_pool2d.default,
            exir_ops.edge.aten._softmax.default,
            exir_ops.edge.aten.view_copy.default,
            exir_ops.edge.aten.clone.default,
            operator.getitem,
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        ]
        return supported


@final
class ArmPartitioner(Partitioner):
    compile_spec = []

    def __init__(self) -> None:
        self.delegation_spec = DelegationSpec(ArmBackend.__name__, self.compile_spec)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        # Run the CapabilityBasedPartitioner to return the largest possible
        # subgraphs containing the nodes with the tags
        logger.info("ArmPartitioner::partition")
        partition_tags = {}

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            TOSASupportedOperators(),
            allows_single_node_partition=True,
        )
        partition_list = capability_partitioner.propose_partitions()
        for partition in partition_list:
            for node in partition.nodes:
                tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = tag
                partition_tags[tag] = self.delegation_spec

        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )
