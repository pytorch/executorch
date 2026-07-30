# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
from executorch.examples.qualcomm.oss_scripts.llama.inference import ModelInference
from torch.utils.data import DataLoader


class QuantizationStrategy(ABC):
    """Base class for quantization strategies."""

    def __init__(
        self,
        inference: ModelInference,
        module: torch.fx.GraphModule,
        tok_embedding: Optional[torch.fx.GraphModule] = None,
    ):
        self._inference = inference
        self._module = module
        self._tok_embedding = tok_embedding

    @abstractmethod
    def _calibrate(self, calib_loader: Dict[str, DataLoader]) -> None: ...

    @abstractmethod
    def quantize(
        self, calib_loader: Dict[str, DataLoader], **kwargs
    ) -> torch.fx.GraphModule: ...
