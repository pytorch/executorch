# Copyright (c) 2024 MediaTek Inc.
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

import torch

from typing import final, Callable, List, Optional, Tuple
from executorch.backends.mediatek.preprocess import NeuropilotBackend
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


def _get_node_type(node: torch.fx.Node) -> str:
    op = node.target
    try:
        op_name = op.name()
    except AttributeError:
        # built-in function getitem
        op_name = op.__name__ if callable(op) else str(op)
    return op_name


class NeuropilotOperatorsSupport(OperatorSupportBase):

    def __init__(self, ops_to_skip: Optional[set] = None) -> None:
        if ops_to_skip is None:
            ops_to_skip = set()

        self._ops_to_skip = ops_to_skip

    def is_node_supported(self, _, node: torch.fx.Node) -> bool:
        # Handle 'call_function' only cause 'placeholder' and 'output' cannot be tagged.
        # Ref: https://github.com/pytorch/executorch/pull/1398
        if node.op != "call_function":
            return False

        op_type = _get_node_type(node)
        if op_type in self._ops_to_skip:
            print(f'[Neuropilot Backend] The {op_type} operator is skipped.')
            return False

        return importer_v2.is_fx_node_supported(node)


@final
class NeuropilotPartitioner(Partitioner):

    def __init__(self, compile_spec: List[CompileSpec], ops_to_skip: Optional[set] = None) -> None:
        self.delegation_spec = DelegationSpec(NeuropilotBackend.__name__, compile_spec)
        self._ops_to_skip = ops_to_skip

    def ops_to_not_decompose(
        self, ep: ExportedProgram,
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        ops_not_decompose = [
            torch.ops.aten.pixel_shuffle.default,
            torch.ops.aten.upsample_bilinear2d.default,
            torch.ops.aten.upsample_bilinear2d.vec,
            torch.ops.aten.upsample_nearest2d.default,
            torch.ops.aten.upsample_nearest2d.vec,
        ]
        return (ops_not_decompose, None)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            NeuropilotOperatorsSupport(self._ops_to_skip),
            allows_single_node_partition=True,
        )
        partition_list = capability_partitioner.propose_partitions()

        partition_tags = {}
        for partition in partition_list:
            for node in partition.nodes:
                tag = f'tag{partition.id}'
                node.meta['delegation_tag'] = tag
                partition_tags[tag] = self.delegation_spec

        tag_constant_data(exported_program)

        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )
