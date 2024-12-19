# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import logging
import os
from typing import Callable, final, List, Optional, Tuple

import torch
from executorch.backends.arm.arm_backend import ArmBackend  # usort: skip
from executorch.backends.arm._passes.tag_io_quant_pass import TagIOQuantPass
from executorch.backends.arm.operator_support.tosa_supported_operators import (
    TOSASupportedOperators,
)
from executorch.backends.arm.tosa_specification import TosaSpecification
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data
from executorch.exir.passes import PassManager
from torch.export.exported_program import ExportedProgram
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
TOSA_DBG_VERBOSE = os.environ.get("TOSA_DBG_VERBOSE") == "1"
if TOSA_DBG_VERBOSE:
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)


@final
class ArmPartitioner(Partitioner):
    def __init__(self, compile_spec: List[CompileSpec]) -> None:
        self.delegation_spec = DelegationSpec(ArmBackend.__name__, compile_spec)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        # Run the CapabilityBasedPartitioner to return the largest possible
        # subgraphs containing the nodes with the tags
        logger.info("ArmPartitioner::partition")
        partition_tags = {}

        tosa_spec = TosaSpecification.create_from_compilespecs(
            self.delegation_spec.compile_specs
        )

        logger.info(f"Partitioning for {tosa_spec}")

        for spec in self.delegation_spec.compile_specs:
            if spec.key == "quantize_io" and spec.value.decode() == "True":
                # Exclude IO quantization from the partition
                passes = PassManager(
                    passes=[
                        TagIOQuantPass(),
                    ]
                )
                passes(exported_program.graph_module)

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            TOSASupportedOperators(tosa_spec),
            allows_single_node_partition=True,
        )
        partition_list = capability_partitioner.propose_partitions()
        for partition in partition_list:
            for node in partition.nodes:
                tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = tag
                partition_tags[tag] = self.delegation_spec

        tag_constant_data(exported_program)

        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )

    def ops_to_not_decompose(
        self,
        ep: ExportedProgram,
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        ops_to_not_decompose = [
            torch.ops.aten.linear.default,
            torch.ops.aten.upsample_nearest2d.vec,
        ]
        return (ops_to_not_decompose, None)
