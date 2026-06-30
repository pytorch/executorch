# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.cortex_m.target_config import CortexMTargetConfig
from executorch.exir.pass_base import ExportPass
from torch.export import ExportedProgram


class CortexMPass(ExportPass):
    """Base class for passes that need the Cortex-M target config.

    Passes that subclass this declare `exported_program` and `target_config`
    in their `__init__`; `CortexMPassManager.transform()` injects both
    automatically when running the pass list.
    """

    def __init__(
        self,
        exported_program: ExportedProgram,
        target_config: CortexMTargetConfig,
    ) -> None:
        super().__init__()
        self._exported_program = exported_program
        self._target_config = target_config

    @property
    def exported_program(self) -> ExportedProgram:
        return self._exported_program

    @property
    def target_config(self) -> CortexMTargetConfig:
        return self._target_config
