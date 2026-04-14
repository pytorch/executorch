# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.exir.pass_base import ExportPass
from torch.export import ExportedProgram

from .cortex_m_configuration import CortexMConfiguration


class CortexMPass(ExportPass):
    """
    An abstract interface for CortexM backend passes.
    """

    def __init__(
        self, exported_program: ExportedProgram, cortex_m_config: CortexMConfiguration
    ) -> None:
        super().__init__()
        self._exported_program = exported_program
        self._cortex_m_config = cortex_m_config

    @property
    def exported_program(self) -> ExportedProgram:
        return self._exported_program

    @property
    def cortex_m_config(self) -> CortexMConfiguration:
        return self._cortex_m_config
