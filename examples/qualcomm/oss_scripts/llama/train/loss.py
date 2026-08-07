# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from executorch.examples.qualcomm.oss_scripts.llama.dataset.constants import (
    LABEL_IGNORE_INDEX,
)


class CrossEntropyLoss:
    """Standard next-token CE loss.

    Args:
        logits: [B, T, V] float tensor.
        labels: [B, T] int64 next-token labels; LABEL_IGNORE_INDEX positions are masked.

    Returns:
        Scalar loss tensor.
    """

    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels[:, :-1].reshape(-1),
            ignore_index=LABEL_IGNORE_INDEX,
        )


class KLDivergenceLoss:
    """Hinton KL-divergence with temperature. Only active (non-LABEL_IGNORE_INDEX) positions.

    Args:
        student_logits: [B, T, V] float tensor, grad attached.
        teacher_logits: [B, T, V] float tensor, no grad.
        labels:         [B, T] int64 next-token labels; LABEL_IGNORE_INDEX is masked.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def compute(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        valid_mask = labels[:, :-1] != LABEL_IGNORE_INDEX
        valid_s = student_logits[:, :-1][valid_mask]
        valid_t = teacher_logits[:, :-1][valid_mask]
        if valid_s.shape[0] == 0:
            return torch.zeros(1, device=student_logits.device).squeeze()
        return F.kl_div(
            F.log_softmax(valid_s / self.temperature, dim=-1),
            F.softmax(valid_t / self.temperature, dim=-1),
            reduction="batchmean",
        ) * (self.temperature * self.temperature)
