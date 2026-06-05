# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class DataConfig:
    """Data pipeline config."""

    max_context_len: int
    token_dtype: Optional[str] = None
    batch_size: int = 1
    seed: int = 42

    # PTQ calibration (also used as QAT Mode 2 calib source)
    calib_tasks: Optional[List[str]] = None
    calib_limit: int = 1
    calib_num_fewshot: Optional[int] = None
    calib_samples: Optional[List[str]] = field(default_factory=list)
    calib_hf_dataset: Optional[str] = None
    calib_hf_limit: int = 1

    # QAT Mode 1 — full split (single dataset pool, auto-split into calib/train/val)
    qat_full_tasks: Optional[List[str]] = None
    qat_full_hf_dataset: Optional[str] = None
    qat_full_hf_limit: int = 200
    qat_full_limit: int = 200
    calib_train_ratio: float = 0.2  # fraction of full pool for calib; rest → train+val

    # QAT Mode 2 — explicit split (separate calib and train sources)
    train_tasks: Optional[List[str]] = None
    train_limit: int = 180
    train_hf_dataset: Optional[str] = None
    train_hf_limit: int = 180

    # Shared by Mode 1 & 2: fraction of the train+val pool used for training
    train_val_ratio: float = 1.0  # 1.0 disables validation

    @property
    def qat_mode(self) -> str:
        """'full' if a single pool is provided for auto-splitting; 'explicit' otherwise."""
        if self.qat_full_tasks or self.qat_full_hf_dataset:
            return "full"
        return "explicit"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "DataConfig":
        return cls(
            max_context_len=args.max_context_len,
            token_dtype=(
                torch.int64
                if getattr(args, "embedding_quantize", None)
                else torch.int32
            ),
            batch_size=getattr(args, "batch_size", 1),
            calib_tasks=getattr(args, "calib_tasks", None),
            calib_limit=getattr(args, "calib_limit", 1),
            calib_num_fewshot=getattr(args, "calib_num_fewshot", None),
            calib_samples=getattr(args, "calib_samples", None) or [],
            calib_hf_dataset=getattr(args, "calib_hf_dataset", None),
            calib_hf_limit=getattr(args, "calib_hf_limit", 1),
            qat_full_tasks=getattr(args, "qat_full_tasks", None),
            qat_full_hf_dataset=getattr(args, "qat_full_hf_dataset", None),
            qat_full_hf_limit=getattr(args, "qat_full_hf_limit", 200),
            qat_full_limit=getattr(args, "qat_full_limit", 200),
            calib_train_ratio=args.calib_train_ratio,
            train_tasks=getattr(args, "train_tasks", None),
            train_limit=getattr(args, "train_limit", 180),
            train_hf_dataset=getattr(args, "train_hf_dataset", None),
            train_hf_limit=getattr(args, "train_hf_limit", 180),
            train_val_ratio=args.train_val_ratio,
        )
