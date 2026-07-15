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
    # calibration
    calib_tasks: Optional[List[str]] = None
    calib_limit: int = 1
    calib_num_fewshot: Optional[int] = None
    calib_samples: Optional[List[str]] = field(default_factory=list)
    batch_size: int = 1

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "DataConfig":
        return cls(
            max_context_len=args.max_context_len,
            token_dtype=(
                torch.int64
                if getattr(args, "embedding_quantize", None)
                else torch.int32
            ),
            calib_tasks=getattr(args, "calib_tasks", None),
            calib_limit=getattr(args, "calib_limit", 1),
            calib_num_fewshot=getattr(args, "calib_num_fewshot", None),
            calib_samples=getattr(args, "calib_samples", None) or [],
            batch_size=getattr(args, "batch_size", 1),
        )
