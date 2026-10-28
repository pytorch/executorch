# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluation purpose adapters (protocol + implementation)."""

from executorch.backends.qualcomm.genai_pipeline.datasets.evaluation.default_evaluation_data_adapter import (
    DefaultEvaluationDataAdapter,
)
from executorch.backends.qualcomm.genai_pipeline.datasets.evaluation.evaluation_data_adapter import (
    EvaluationDataAdapter,
)

__all__ = [
    "EvaluationDataAdapter",
    "DefaultEvaluationDataAdapter",
]
