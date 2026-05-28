# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from dataclasses import dataclass, fields
from enum import auto, Enum
from typing import Any


class SoftmaxDecompositionConfig(Enum):
    MASKED = auto()  # Stable softmax + masked fill decomposition
    STABLE = auto()  # Stable softmax, no masked fill decomposition


@dataclass
class ArmPassPipelineConfig:
    softmax: SoftmaxDecompositionConfig = SoftmaxDecompositionConfig.MASKED

    def is_default(self) -> bool:
        return self.softmax is SoftmaxDecompositionConfig.MASKED

    def to_dict(self) -> dict[str, str]:
        return {f.name: getattr(self, f.name).name for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArmPassPipelineConfig":
        config = cls()
        for f in fields(cls):
            raw_value = data.get(f.name)
            if raw_value is None:
                continue
            enum_type = f.type
            setattr(config, f.name, enum_type[raw_value])
        return config

    def serialize(self) -> bytes:
        """Return a serialized representation of this config."""
        return json.dumps(self.to_dict()).encode()

    def __repr__(self):
        fields = ", ".join(f"{name}={value!r}" for name, value in self.__dict__.items())
        return f"({fields})"
