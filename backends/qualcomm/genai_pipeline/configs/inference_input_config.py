# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
    from torch import nn


@dataclass
class InferenceInputConfig:
    """Input configuration for the inference stage.

    Attributes:
        soc_model: The target SoC (e.g., QcomChipset.SM8750). Required.
        artifact_dir: Directory for inference outputs and runtime files.
            Required.
        pte_paths: Compiled ``.pte`` files keyed by artifact name (see
            ``artifact_keys``), as produced by the compilation stage. ``None``
            when compilation was skipped, in which case the inference strategy
            resolves them from ``--pre-gen-pte``. The runner passes each to a
            different argument, so they are addressed by name rather than by
            position.
        tokenizer: The tokenizer instance for encoding/decoding.
        runtime_tokenizer_path: Path to runtime tokenizer for on-device use.
        prompt: The user prompt(s) for text generation.
        model_module: Model preparation's FP32 modules, keyed by component, for
            evaluators that score the runner against the host model. ``None``
            when model preparation was skipped.
        example_inputs: Per-component, per-graph positional example inputs of
            ``model_module``, which a host forward needs for the static graph's
            attention mask and KV-cache placeholders. ``None`` when model
            preparation was skipped.
    """

    soc_model: "QcomChipset"
    artifact_dir: Path
    pte_paths: Optional[Dict[str, Path]] = None
    tokenizer: Any = None
    runtime_tokenizer_path: Optional[Path] = None
    prompt: Optional[List[str]] = None
    model_module: Optional[Dict[str, "nn.Module"]] = None
    example_inputs: Optional[Dict[str, Dict[str, Tuple[Any, ...]]]] = None
