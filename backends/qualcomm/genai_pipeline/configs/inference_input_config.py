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
        artifact_paths: Compiled model artifacts (.pte files), keyed by artifact
            name (see ``artifact_keys``), as produced by the compilation stage.
            The on-device runner passes each to a different argument, so they
            are addressed by name rather than by position.
        tokenizer: The tokenizer instance for encoding/decoding.
        runtime_tokenizer_path: Path to runtime tokenizer for on-device use.
        prompt: The user prompt(s) for text generation.
        inference_options: Engine-specific inference options.
    """

    soc_model: "QcomChipset"
    artifact_paths: Optional[Dict[str, Path]] = None
    tokenizer: Any = None
    runtime_tokenizer_path: Optional[Path] = None
    prompt: Optional[List[str]] = None
    inference_options: Dict[str, Any] = field(default_factory=dict)
