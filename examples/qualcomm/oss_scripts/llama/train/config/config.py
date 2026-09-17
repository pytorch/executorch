# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import get_type_hints, Optional

import yaml


@dataclass
class TrainingArgs:
    """Training algorithm hyperparameters. Model structure and data pipeline concerns live elsewhere."""

    epochs: int = 1
    lr: float = 2e-5
    alpha: float = 0.99  # KD weight; (1 - alpha) is CE weight
    temperature: float = 1.0
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.05
    # Path to a YAML for per-param-group
    # LR (see train/config/lr_config.yaml). Set from --lr_config, not from
    # --train_config, so it is not subject to the type conversion below.
    lr_config: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingArgs":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        field_types = get_type_hints(cls)
        unknown = raw.keys() - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(
                f"Unknown TrainingArgs field(s) in {path}: {sorted(unknown)}"
            )
        converted = {
            k: field_types[k](v) if field_types[k] in (int, float) else v
            for k, v in raw.items()
        }
        return cls(**converted)
