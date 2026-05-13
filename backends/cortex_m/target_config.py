# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from enum import auto, Enum
from typing import Optional

import cmsis_nn  # type: ignore[import-not-found, import-untyped]


class CortexM(Enum):
    """Cortex-M CPU variant. Names mirror cmsis_nn.CortexM so the cmsis_nn
    enum can be looked up by name."""

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


# Per-CPU set of cmsis_nn backends the core can execute. SCALAR is
# universal; DSP requires the Armv7E-M or Armv8-M-Mainline DSP option;
# MVE requires Armv8.1-M Mainline with the MVE extension. The supersession
# (SCALAR < DSP < MVE) reflects that an MVE-capable core also runs DSP
# and scalar code, which is what makes "M55 without MVE" → DSP override
# legitimate.
_SUPPORTED_BACKENDS: dict[CortexM, frozenset[cmsis_nn.Backend]] = {
    CortexM.M0: frozenset({cmsis_nn.Backend.SCALAR}),
    CortexM.M0PLUS: frozenset({cmsis_nn.Backend.SCALAR}),
    CortexM.M3: frozenset({cmsis_nn.Backend.SCALAR}),
    CortexM.M23: frozenset({cmsis_nn.Backend.SCALAR}),
    CortexM.M4: frozenset({cmsis_nn.Backend.SCALAR, cmsis_nn.Backend.DSP}),
    CortexM.M7: frozenset({cmsis_nn.Backend.SCALAR, cmsis_nn.Backend.DSP}),
    CortexM.M33: frozenset({cmsis_nn.Backend.SCALAR, cmsis_nn.Backend.DSP}),
    CortexM.M35P: frozenset({cmsis_nn.Backend.SCALAR, cmsis_nn.Backend.DSP}),
    CortexM.M55: frozenset(
        {cmsis_nn.Backend.SCALAR, cmsis_nn.Backend.DSP, cmsis_nn.Backend.MVE}
    ),
    CortexM.M85: frozenset(
        {cmsis_nn.Backend.SCALAR, cmsis_nn.Backend.DSP, cmsis_nn.Backend.MVE}
    ),
}


@dataclass(frozen=True)
class CortexMTargetConfig:
    """AOT compile target configuration for the Cortex-M backend.

    `cpu` selects the CPU variant. `isa` optionally overrides the cmsis_nn
    backend that would normally be derived from `cpu` — useful for cores
    with optional ISA extensions (M55 without MVE, M33 without DSP, etc.).
    Overrides are validated against the CPU's architectural capability set
    on construction; e.g. forcing MVE on an M0 raises ValueError.
    """

    cpu: CortexM
    isa: Optional[cmsis_nn.Backend] = None

    def __post_init__(self) -> None:
        if self.isa is None:
            return
        supported = _SUPPORTED_BACKENDS.get(self.cpu)
        if supported is None or self.isa not in supported:
            allowed = sorted(b.name for b in supported) if supported else []
            raise ValueError(
                f"Backend {self.isa.name} is not supported on "
                f"{self.cpu.name}; supported: {allowed}"
            )

    @property
    def backend(self) -> cmsis_nn.Backend:
        if self.isa is not None:
            return self.isa
        try:
            cmsis_member = getattr(cmsis_nn.CortexM, self.cpu.name)
        except AttributeError as e:
            raise ValueError(
                f"cmsis_nn does not yet support {self.cpu.name}; pass an "
                f"explicit `isa=` override or wait for upstream support."
            ) from e
        return cmsis_nn.resolve_backend(cmsis_member)

    @classmethod
    def from_target_string(cls, target: str) -> CortexMTargetConfig:
        """Parse a `cortex-m<variant>` target string."""
        if not target.startswith("cortex-m"):
            raise ValueError(
                f"Cortex-M target string must start with 'cortex-m', "
                f"got: {target!r}"
            )
        enum_name = "M" + target[len("cortex-m") :].upper()
        try:
            cpu = CortexM[enum_name]
        except KeyError as e:
            raise ValueError(
                f"Unsupported Cortex-M target string: {target!r}. "
                f"Supported: {sorted('cortex-m' + m.name[1:].lower() for m in CortexM)}"
            ) from e
        return cls(cpu=cpu)
