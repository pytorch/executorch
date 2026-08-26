# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass

from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.backends.cortex_m.target_config import CortexMTargetConfig
from executorch.exir.backend.op_backend import OpBackend
from torch.export import ExportedProgram


@dataclass(frozen=True)
class CortexMOpBackend(OpBackend):
    """Lowers the graph to CMSIS-NN kernels.

    Each pass is built against the previous pass's output, so the pass manager
    drives the sequence rather than handing back a pass list.
    """

    target_config: CortexMTargetConfig

    def lower(
        self, exported_program: ExportedProgram, method_name: str
    ) -> ExportedProgram:
        return CortexMPassManager(
            exported_program, target_config=self.target_config
        ).transform()
