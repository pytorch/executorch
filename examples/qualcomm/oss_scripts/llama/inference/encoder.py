# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch


class EncoderInference:
    """Runs a vision/audio encoder; returns a tuple of hidden-state tensors."""

    @torch.no_grad()
    def predict_step(
        self,
        module: torch.nn.Module,
        inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        result = module(*(inputs,))
        return result
