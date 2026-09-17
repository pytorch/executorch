# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


class EngineType(Enum):
    """Selects the inference framework for a pipeline stage."""

    EXECUTORCH = "executorch"


# Pipeline stage names
STAGE_MODEL_PREPARATION = "model_preparation"
STAGE_QUANTIZATION = "quantization"
STAGE_COMPILATION = "compilation"
STAGE_INFERENCE = "inference"

ALL_STAGES = frozenset(
    {
        STAGE_MODEL_PREPARATION,
        STAGE_QUANTIZATION,
        STAGE_COMPILATION,
        STAGE_INFERENCE,
    }
)
