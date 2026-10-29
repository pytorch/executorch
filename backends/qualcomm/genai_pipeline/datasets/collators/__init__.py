# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Component-aware collators, one per modality.

Each collator maps a model component to its collate function (padding, mask
construction). Modality-specific: ``LLMDatasetCollector`` covers the text
decoder alone; ``MLLMDatasetCollector`` adds the vision / audio encoders.
"""

from executorch.backends.qualcomm.genai_pipeline.datasets.collators.llm_collator import (
    LLMDatasetCollector,
)
from executorch.backends.qualcomm.genai_pipeline.datasets.collators.mllm_collator import (
    MLLMDatasetCollector,
)

__all__ = [
    "LLMDatasetCollector",
    "MLLMDatasetCollector",
]
