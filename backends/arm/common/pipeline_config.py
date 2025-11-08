# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from dataclasses import dataclass, fields
from enum import auto, Enum
from typing import Any


class SoftmaxDecompositionConfig(Enum):
    MASKED = auto()
    UNSTABLE = auto()


class FuseDuplicateUsersConfig(Enum):
    ENABLED = auto()
    DISABLED = auto()


@dataclass
class ArmPassPipelineConfig:
    softmax: SoftmaxDecompositionConfig = SoftmaxDecompositionConfig.MASKED
    fuse_duplicate_users: FuseDuplicateUsersConfig = FuseDuplicateUsersConfig.ENABLED

    def disable_masked_softmax(self) -> None:
        self.softmax = SoftmaxDecompositionConfig.UNSTABLE

    def disable_fuse_duplicate_users(self) -> None:
        self.fuse_duplicate_users = FuseDuplicateUsersConfig.DISABLED

    def is_default(self) -> bool:
        return (
            self.softmax is SoftmaxDecompositionConfig.MASKED
            and self.fuse_duplicate_users is FuseDuplicateUsersConfig.ENABLED
        )

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
