# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn


@dataclass
class ModelPreparationOutputConfig:
    """Output produced by the model preparation stage.

    Attributes:
        model_module: The prepared nn.Module ready for quantization.
        tokenizer: The tokenizer instance for encoding/decoding text.
        calibration_data: Dataset samples for calibration during quantization.
        runtime_tokenizer_path: Path to runtime tokenizer for on-device inference.
        chat_template: Optional chat template for instruct models.
    """

    model_module: Optional["nn.Module"] = None
    tokenizer: Any = None
    calibration_data: Optional[List[Any]] = None
    runtime_tokenizer_path: Optional[Path] = None
    chat_template: Optional[str] = None
