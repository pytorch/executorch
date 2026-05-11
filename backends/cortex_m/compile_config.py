# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Cpu = Literal[
    "cortex-m0",
    "cortex-m0plus",
    "cortex-m3",
    "cortex-m4",
    "cortex-m7",
    "cortex-m23",
    "cortex-m33",
    "cortex-m35p",
    "cortex-m52",
    "cortex-m55",
    "cortex-m85",
]
Isa = Literal["scalar", "dsp", "mve"]

# Default ISA per CPU follows the most common configuration each core is
# shipped with. M33/M35P optionally lack DSP, and M52/M55/M85 optionally
# lack MVE; callers can pass `isa=` explicitly to override.
_CPU_DEFAULT_ISA: dict[str, str] = {
    "cortex-m0": "scalar",
    "cortex-m0plus": "scalar",
    "cortex-m3": "scalar",
    "cortex-m4": "dsp",
    "cortex-m7": "dsp",
    "cortex-m23": "scalar",
    "cortex-m33": "dsp",
    "cortex-m35p": "dsp",
    "cortex-m52": "mve",
    "cortex-m55": "mve",
    "cortex-m85": "mve",
}

_SUPPORTED_FEATURES: frozenset[str] = frozenset({"int8"})


@dataclass(frozen=True)
class CortexMCompileConfig:
    """AOT compile configuration for the Cortex-M backend.

    `cpu` and `isa` are consumed by passes that need to differ by target — most
    notably any future AOT scratch-buffer sizing — and threaded through the
    build system as the `-mcpu=` value.

    The current default matches pre-config behavior (M55 + MVE) so callers that
    don't opt in see no change.
    """

    cpu: Cpu = "cortex-m55"
    isa: Isa | None = None

    def __post_init__(self) -> None:
        if self.cpu not in _CPU_DEFAULT_ISA:
            raise ValueError(
                f"Unsupported Cortex-M CPU: {self.cpu!r}. "
                f"Supported: {sorted(_CPU_DEFAULT_ISA)}"
            )
        if self.isa is None:
            # frozen dataclass: use object.__setattr__ to fill default ISA.
            object.__setattr__(self, "isa", _CPU_DEFAULT_ISA[self.cpu])

    @classmethod
    def from_target_string(cls, target: str) -> CortexMCompileConfig:
        """Parse `cortex-m<variant>+int8` strings used by `aot_arm_compiler.py`.

        Today only `+int8` is supported. The suffix is required so the target
        string remains explicit about the data type contract.
        """
        cpu, sep, features = target.partition("+")
        if not sep:
            raise ValueError(
                f"Cortex-M target string must include a feature suffix "
                f"(e.g. '+int8'), got: {target!r}"
            )
        feature_set = set(features.split("+"))
        unknown = feature_set - _SUPPORTED_FEATURES
        if unknown or "int8" not in feature_set:
            raise ValueError(
                f"Cortex-M target string must be '<cpu>+int8' "
                f"(supported features: {sorted(_SUPPORTED_FEATURES)}), "
                f"got: {target!r}"
            )
        if cpu not in _CPU_DEFAULT_ISA:
            raise ValueError(
                f"Unsupported Cortex-M CPU in target string: {cpu!r}. "
                f"Supported: {sorted(_CPU_DEFAULT_ISA)}"
            )
        return cls(cpu=cpu)  # type: ignore[arg-type]
