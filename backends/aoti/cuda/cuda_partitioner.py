# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Callable, Dict, final, List, Optional, Tuple

import torch
from executorch.backends.aoti.cuda.cuda_backend import CudaBackend  # usort: skip
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data
from torch.export.exported_program import ExportedProgram


@final
class CudaPartitioner(Partitioner):
    def __init__(self, compile_spec: List[CompileSpec]) -> None:
        self.delegation_spec = DelegationSpec(CudaBackend.__name__, compile_spec)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        """
        Fully delegate the graph to AOTInductor by tagging all nodes as a single partition.
        """

        partition_tags: Dict[str, DelegationSpec] = {}
        for node in exported_program.graph.nodes:
            if node.op != "call_function":
                continue
            tag = f"tag0"
            node.meta["delegation_tag"] = tag
            partition_tags[tag] = self.delegation_spec

        tag_constant_data(exported_program)

        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )

    def ops_to_not_decompose(
        self, ep: ExportedProgram
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        """
        Return a list of operations that should not be decomposed and let the AOT compiler handle them.
        Currently we skip decomposing all ops and let the AOT compiler handle them.
        """
        do_not_decompose = set()

        for node in ep.graph.nodes:
            if node.op == "call_function" and isinstance(
                node.target, torch._ops.OpOverload
            ):
                do_not_decompose.add(node.target)
        return list(do_not_decompose), None
