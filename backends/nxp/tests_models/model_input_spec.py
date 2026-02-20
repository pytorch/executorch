# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import numpy as np
import torch
from torch import memory_format


@dataclass
class ModelInputSpec:
    shape: tuple[int, ...]
    type: np.dtype = np.float32
    dtype: torch.dtype = torch.float32
    dim_order: memory_format = torch.contiguous_format
