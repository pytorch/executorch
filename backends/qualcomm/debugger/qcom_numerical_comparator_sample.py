# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import executorch.exir as exir
import torch
from executorch.backends.qualcomm.debugger.qcom_numerical_comparator_base import (
    QcomNumericalComparatorBase,
)


"""
This file provides some examples on how to implement a QcomNumericalComparator
"""


class QcomMSEComparator(QcomNumericalComparatorBase):
    """Mean Squared Error comparator for Qualcomm intermediate outputs."""

    def __init__(self, edge_ep: exir.ExportedProgram, threshold: float = 1e-3) -> None:
        super().__init__(edge_ep)
        self.threshold = threshold

    def metric_name(self) -> str:
        return "mse"

    def is_valid_score(self, score: float) -> bool:
        return score <= self.threshold

    def element_compare(self, a: Any, b: Any) -> float:
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return torch.mean(torch.square(a.float() - b.float())).item()
        return float((a - b) ** 2)


class QcomCosineSimilarityComparator(QcomNumericalComparatorBase):
    """Cosine Similarity comparator for Qualcomm intermediate outputs."""

    def __init__(self, edge_ep: exir.ExportedProgram, threshold: float = 0.95) -> None:
        super().__init__(edge_ep)
        self.threshold = threshold

    def metric_name(self) -> str:
        return "cosine_similarity"

    def is_valid_score(self, score: float) -> bool:
        return score >= self.threshold

    def element_compare(self, a: Any, b: Any) -> float:
        score = torch.nn.functional.cosine_similarity(
            a.to(torch.float32).flatten(), b.to(torch.float32).flatten(), dim=0
        ).item()
        return score
