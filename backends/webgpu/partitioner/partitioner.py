# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Callable, Dict, final, List, Optional, Tuple

import torch
from executorch.backends.vulkan.op_registry import OpKey
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir.backend.partitioner import Partitioner, PartitionResult
from torch.export import ExportedProgram


@final
class WebGPUPartitioner(Partitioner):
    """WebGPU frontend for the Vulkan partitioning and serialization path."""

    def __init__(
        self,
        compile_options: Optional[Dict[str, Any]] = None,
        operator_blocklist: Optional[List[OpKey]] = None,
        operator_allowlist: Optional[List[OpKey]] = None,
        nn_module_blocklist: Optional[List[str]] = None,
        nn_module_allowlist: Optional[List[str]] = None,
    ) -> None:
        self._vulkan_partitioner = VulkanPartitioner(
            compile_options=compile_options,
            operator_blocklist=operator_blocklist,
            operator_allowlist=operator_allowlist,
            nn_module_blocklist=nn_module_blocklist,
            nn_module_allowlist=nn_module_allowlist,
        )

    def ops_to_not_decompose(
        self, ep: ExportedProgram
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        return self._vulkan_partitioner.ops_to_not_decompose(ep)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        return self._vulkan_partitioner.partition(exported_program)
