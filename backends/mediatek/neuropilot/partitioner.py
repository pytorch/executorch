# Copyright (c) 2024 MediaTek Inc.
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

import torch

from typing import final, List
from executorch.backends.mediatek.neuropilot.preprocess import NeuropilotBackend
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data
from torch.export.exported_program import ExportedProgram
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase

from mtk_converter.python.converters.pytorch import importer_v2


class NeuropilotOperatorsSupport(OperatorSupportBase):

    def is_node_supported(self, _, node: torch.fx.Node) -> bool:
        # Handle 'call_function' only cause 'placeholder' and 'output' cannot be tagged.
        # Ref: https://github.com/pytorch/executorch/pull/1398
        if node.op != "call_function":
            return False

        # TODO: Add mechansim to ignore some specific nodes
        is_supported = importer_v2.is_fx_node_supported(node)
        return is_supported


@final
class NeuropilotPartitioner(Partitioner):

    def __init__(self, compile_spec: List[CompileSpec]) -> None:
        self.delegation_spec = DelegationSpec(NeuropilotBackend.__name__, compile_spec)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        # Run the CapabilityBasedPartitioner to return the largest possible
        partition_tags = {}

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            NeuropilotOperatorsSupport(),
            allows_single_node_partition=True,
        )
        partition_list = capability_partitioner.propose_partitions()
        for partition in partition_list:
            for node in partition.nodes:
                tag = f'tag{partition.id}'
                node.meta['delegation_tag'] = tag
                partition_tags[tag] = self.delegation_spec

        tag_constant_data(exported_program)

        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )
