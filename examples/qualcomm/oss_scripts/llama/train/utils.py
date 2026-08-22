# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import fnmatch
import logging
import math
from typing import Dict, List, Optional

import torch
import yaml


def get_warmup_cosine_lr(
    optimizer: torch.optim.Optimizer,
    warmup_min_lr: float,
    warmup_max_lr: float,
    warmup_num_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up from warmup_min_lr to warmup_max_lr, then cosine decay to 0.

    Matches HuggingFace's `lr_scheduler_type: cosine` (single half-cycle, no min-lr floor).
    """
    min_ratio = warmup_min_lr / warmup_max_lr
    warmup_slope = (1.0 - min_ratio) / max(1, warmup_num_steps)
    decay_steps = max(1, total_steps - warmup_num_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_num_steps:
            return min_ratio + step * warmup_slope
        progress = min(1.0, (step - warmup_num_steps) / decay_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _numeric_str_to_float(value):
    # PyYAML parses bare scientific notation (e.g. "4e-6") as a string, not a
    # float, unless it has a decimal point (e.g. "4.0e-6"). Convert explicitly so
    # an lr_config entry like `lr: 4e-6` doesn't silently reach the optimizer as
    # a str (AdamW accepts a str lr at construction time but crashes on .step()).
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def build_param_groups(
    model: torch.nn.Module, lr_config_path: Optional[str]
) -> List[Dict]:
    """Split params into AdamW groups, giving each group its own kwargs.

    lr_config_path is a YAML file mapping glob patterns to per-group
    optimizer kwargs (see train/config/lr_config.yaml), matched against
    `model.named_parameters()` names.
    """
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    pattern_cfg = {}
    if lr_config_path:
        with open(lr_config_path) as f:
            pattern_cfg = yaml.safe_load(f) or {}

    used = set()
    groups = []
    for pattern, overrides in pattern_cfg.items():
        matched = [
            (n, p)
            for n, p in trainable
            if n not in used and fnmatch.fnmatchcase(n, pattern)
        ]
        if not matched:
            continue
        used.update(n for n, _ in matched)
        group = {k: _numeric_str_to_float(v) for k, v in overrides.items()}
        if "betas" in group:
            group["betas"] = tuple(group["betas"])
        group["params"] = [p for _, p in matched]
        groups.append(group)
        logging.info(
            f"lr_config: pattern '{pattern}' matched {len(matched)} param(s) -> {overrides}"
        )

    default_params = [p for n, p in trainable if n not in used]
    if default_params:
        groups.append({"params": default_params})
        logging.info(
            f"lr_config: {len(default_params)} param(s) using default optimizer settings"
        )

    return groups
