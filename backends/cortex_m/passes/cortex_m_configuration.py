# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import auto, Enum
from typing import ClassVar, Mapping

import cmsis_nn  # type: ignore[import-not-found, import-untyped]


class CortexMConfiguration(Enum):
    M0 = auto()
    M0PLUS = auto()
    M3 = auto()
    M4 = auto()
    M7 = auto()
    M23 = auto()
    M33 = auto()
    M35P = auto()
    M55 = auto()
    M85 = auto()
    ANY = auto()  # Guaranteed to work on any Cortex-M.
    __members__: ClassVar[Mapping[str, "CortexMConfiguration"]]

    @property
    def backend(self) -> cmsis_nn.Backend:
        if self == CortexMConfiguration.ANY:
            # Currently, MVE is all we support. We can just return the MVE backend.
            return cmsis_nn.Backend.MVE

        cmsis_nn_cortex_m = cmsis_nn.CortexM.__members__.get(self.name, None)
        if cmsis_nn_cortex_m is None:
            raise ValueError(
                f"CortexM configuration {self.name} is not supported by cmsis_nn."
            )
        return cmsis_nn.resolve_backend(cmsis_nn_cortex_m)
