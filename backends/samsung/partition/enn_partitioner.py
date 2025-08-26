# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import executorch.backends.samsung.python.PyEnnWrapperAdaptor as PyEnnWrapper

import torch
from executorch.backends.samsung.enn_preprocess import ExynosBackend

from executorch.backends.samsung.utils.utils import get_compile_spec
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
from torch.fx.passes.infra.partitioner import Partition
from torch.fx.passes.operator_support import OperatorSupportBase


class EnnOperatorSupport(OperatorSupportBase):

    def __init__(
        self,
        edge_program: torch.export.ExportedProgram,
        compile_specs: List[CompileSpec],
    ):
        self.edge_program = edge_program
        self.enn_wrapper = PyEnnWrapper.EnnWrapper()
        option_spec = get_compile_spec(compile_specs, "Exynos compile", required=True)
        self.enn_wrapper.Init(option_spec.value)

    def is_node_supported(self, _, node: torch.fx.Node) -> bool:
        if node.op != "call_function":
            return False

        if node.op in [
            "get_attr",
            "placeholder",
            "output",
        ]:
            return False

        supported = self.enn_wrapper.IsNodeSupportedByBackend()
        return supported

    def __del__(self):
        self.enn_wrapper.Destroy()


class EnnPartitioner(Partitioner):
    def __init__(self, compile_specs: List[CompileSpec]):
        # TODO(anyone): Add meaningful initialize
        self.delegation_spec = DelegationSpec(ExynosBackend.__name__, compile_specs)
        self.partition_tags: Dict[str, DelegationSpec] = {}
        self.compile_specs = compile_specs

    def generate_partitions(
        self, edge_program: torch.export.ExportedProgram
    ) -> List[Any]:

        return generate_partitions_from_list_of_nodes(
            edge_program.graph_module,
            op_support=self.op_support_checker,
        )

    def tag_nodes(self, partitions: List[Partition]) -> None:
        partition_tags: Dict[str, DelegationSpec] = {}
        for partition in partitions:
            # Add delegation tags
            for node in partition.nodes:
                delegation_tag = f"enn_{partition.id}"
                node.meta["delegation_tag"] = delegation_tag
                partition_tags[delegation_tag] = self.delegation_spec
        return partition_tags

    # override
    def partition(self, edge_program: torch.export.ExportedProgram) -> PartitionResult:
        partitions = self.generate_partitions(edge_program)
        logging.info(f"Find {len(partitions)} " "subgraphs to partition and lowering.")
        if len(partitions) != 0:
            self.partition_tags = self.tag_nodes(partitions)
            tag_constant_data(edge_program)
        del self.op_support_checker
        return PartitionResult(
            tagged_exported_program=edge_program, partition_tags=self.partition_tags
        )
