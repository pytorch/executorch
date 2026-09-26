# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from executorch.devtools.etrecord import ETRecord


@dataclass
class CompilationOutputConfig:
    """Output produced by the compilation stage.

    Attributes:
        artifact_paths: Compiled artifacts (.pte files), keyed by artifact name
            (see ``artifact_keys``). A model may produce several: a text decoder
            always, plus a token embedding and modality encoders when
            multimodal. Keyed rather than ordered because the inference stage
            addresses them individually and the set present varies by model.
            Absent artifacts are omitted rather than mapped to ``None``.
        etrecord: Optional ETRecord for debugging. ExecuTorch engine only.
    """

    artifact_paths: Optional[Dict[str, Path]] = None
    etrecord: Optional["ETRecord"] = None
