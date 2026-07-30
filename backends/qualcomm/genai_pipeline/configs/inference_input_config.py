# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset


@dataclass
class InferenceInputConfig:
    """Input configuration for the inference stage.

    Attributes:
        soc_model: The target SoC (e.g., QcomChipset.SM8750). Required.
        artifact_paths: Paths to compiled model artifacts (.pte files).
            List to support multi-split models (e.g., prefill + decode).
        tokenizer: The tokenizer instance for encoding/decoding.
        runtime_tokenizer_path: Path to runtime tokenizer for on-device use.
        prompt: The user prompt(s) for text generation.
        inference_options: Engine-specific inference options.
    """

    soc_model: "QcomChipset"
    artifact_paths: Optional[List[Path]] = None
    tokenizer: Any = None
    runtime_tokenizer_path: Optional[Path] = None
    prompt: Optional[List[str]] = None
    inference_options: Dict[str, Any] = field(default_factory=dict)
