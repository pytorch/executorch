# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from executorch.backends.qualcomm.serialization.qc_schema import (
        QcomChipset,
        QnnExecuTorchBackendType,
    )
    from executorch.exir.backend.compile_spec import CompileSpec
    from torch import nn


@dataclass
class CompilationInputConfig:
    """Input configuration for the compilation stage.

    Attributes:
        soc_model: The target SoC (e.g., QcomChipset.SM8750). Required.
        backend_type: QNN backend type (HTP, GPU, LPAI, etc.). Required.
        model: The nn.Module to compile (quantized or original for FP16 mode).
        artifact_dir: Directory to store compiled artifacts.
        compile_specs: QNN compiler specifications for backend delegation.
    """

    soc_model: "QcomChipset"
    backend_type: "QnnExecuTorchBackendType"
    model: Optional["nn.Module"] = None
    artifact_dir: Path = field(default_factory=lambda: Path("."))
    compile_specs: Optional[List["CompileSpec"]] = None
