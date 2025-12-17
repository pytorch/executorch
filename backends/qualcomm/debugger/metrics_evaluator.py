# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch


class MetricEvaluatorBase(ABC):
    @abstractmethod
    def metric_name(self) -> str:
        """
        A name for this metric evaluation

        Returns:
            str: name of the metric evaluation
        """
        ...

    @abstractmethod
    def evaluate(
        self, qnn_output: torch.Tensor, cpu_output: torch.Tensor, **kwargs
    ) -> Tuple[Any, bool]:
        """
        This abstract method should accept both QNN and CPU outputs for a single layer.
        Define your own logic to compare the results.

        Args:
            qnn_output (torch.Tensor): QNN intermediate output
            cpu_output (torch.Tensor): CPU intermediate output

        Returns:
            Tuple[Any, bool]: Return 2 elements:
                1) Score or anything that you would like to be printed under metrics category for svg graph or csv file.
                2) A boolean that indicates whether the evaluation result is acceptable or not.
        """
        ...


class AtolEvaluator(MetricEvaluatorBase):
    def __init__(self, threshold=1e-1):
        self.threshold = threshold

    def metric_name(self) -> str:
        return "Atol Similarity"

    def evaluate(
        self, qnn_output: torch.Tensor, cpu_output: torch.Tensor
    ) -> Tuple[Any, bool]:
        avg_atol = torch.mean(torch.abs(qnn_output - cpu_output))
        valid = avg_atol < self.threshold
        formatted_score = f"{avg_atol:.3f}"
        return formatted_score, valid


class CosineSimilarityEvaluator(MetricEvaluatorBase):
    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def metric_name(self) -> str:
        return "Cosine Similarity"

    def evaluate(
        self, qnn_output: torch.Tensor, cpu_output: torch.Tensor
    ) -> Tuple[Any, bool]:
        score = torch.nn.functional.cosine_similarity(
            qnn_output.flatten(), cpu_output.flatten(), dim=0
        ).item()
        valid = score > self.threshold
        formatted_score = f"{score:.3f}"
        return formatted_score, valid


class MeanSquaredErrorEvaluator(MetricEvaluatorBase):
    def __init__(self, threshold=0.01):
        self.threshold = threshold

    def metric_name(self) -> str:
        return "Mean Squared Error"

    def evaluate(
        self, qnn_output: torch.Tensor, cpu_output: torch.Tensor
    ) -> Tuple[Any, bool]:
        mse = torch.mean((qnn_output - cpu_output) ** 2)
        valid = mse < self.threshold
        return mse, valid
