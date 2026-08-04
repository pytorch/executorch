# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ModelPreparationInputConfig:
    """Input configuration for the model preparation stage.

    Attributes:
        model_name: Model identifier (e.g., "llama3_2-1b_instruct"). Required.
        soc_model: Target SoC (e.g., "SM8750"). Required.
        extra_options: Additional model-preparation-specific options.
    """

    model_name: str
    soc_model: str
    extra_options: Dict[str, Any] = field(default_factory=dict)
