# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True, init=False)
class ArmAnnotationInfo(dict):
    """
    Dataclass wrapper that behaves like a dict so serialization can treat it as
    a plain mapping, while still exposing a typed attribute for convenience.
    """

    quantized: bool
    CUSTOM_META_KEY: str = "_arm_annotation_info"

    def __init__(
        self,
        value: Optional[Mapping[str, Any]] = None,
        *,
        quantized: Optional[bool] = None,
    ) -> None:
        if quantized is not None:
            resolved = bool(quantized)

        elif isinstance(value, Mapping):
            resolved = bool(value.get("quantized", False))

        else:
            raise TypeError(
                "ArmAnnotationInfo expects a mapping with a 'quantized' entry or a keyword 'quantized'."
            )
        dict.__init__(self, quantized=resolved)
        object.__setattr__(self, "quantized", resolved)
