# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TensorRT partitioner for ExecuTorch."""

from typing import Dict, List, Optional

from executorch.backends.nvidia.tensorrt.backend import TensorRTBackend
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from torch.export.exported_program import ExportedProgram


class TensorRTPartitioner(Partitioner):
    """Partitioner for TensorRT backend.
    """

    def __init__(
        self,
        compile_specs: Optional[List[CompileSpec]] = None,
    ) -> None:
        super().__init__()
        self.compile_specs = compile_specs or []
        self.delegation_spec = DelegationSpec(
            backend_id=TensorRTBackend.__name__,
            compile_specs=self.compile_specs,
        )

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        """Partition the graph for TensorRT delegation.

        Identifies subgraphs that can be lowered to the TensorRT backend.
        """
        partition_tags: Dict[str, DelegationSpec] = {}
        return PartitionResult(
            tagged_exported_program=exported_program,
            partition_tags=partition_tags,
        )
