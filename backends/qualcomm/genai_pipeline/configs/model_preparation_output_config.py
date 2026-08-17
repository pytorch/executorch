# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn


@dataclass
class ModelPreparationOutputConfig:
    """Output produced by the model preparation stage.

    All fields are ``Optional`` because ``GenAIPipeline._run_model_preparation``
    returns an empty instance when the stage is skipped (e.g. compile-only flows
    with a caller-supplied module). ``None`` is a "stage did not run" sentinel,
    not a valid post-execution state: when the stage runs, ``model_module`` and
    ``tokenizer`` are always populated.

    Attributes:
        model_module: The prepared nn.Module ready for quantization.
        tokenizer: The tokenizer instance for encoding/decoding text.
        calibration_data: Calibration samples. Any Iterable[Tuple[Tensor, ...]],
            including a DataLoader with a custom collate_fn.
        runtime_tokenizer_path: Path to the runtime tokenizer **file** (not the
            containing directory) for on-device inference.
        chat_template: Optional chat template for instruct models.
    """

    model_module: Optional["nn.Module"] = None
    tokenizer: Any = None
    calibration_data: Optional[Iterable[Any]] = None
    runtime_tokenizer_path: Optional[Path] = None
    chat_template: Optional[str] = None
