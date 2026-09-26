# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from executorch.devtools.etrecord import ETRecord


@dataclass
class CompilationOutputConfig:
    """Output produced by the compilation stage.

    Attributes:
        artifact_paths: Paths to the compiled artifacts (.pte files).
            List to support multi-split models where compilation produces
            multiple .pte files (e.g., prefill + decode).
        etrecord: Optional ETRecord for debugging. ExecuTorch engine only.
    """

    artifact_paths: Optional[List[Path]] = None
    etrecord: Optional["ETRecord"] = None
