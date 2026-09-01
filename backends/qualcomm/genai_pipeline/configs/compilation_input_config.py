# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

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

    ``model`` and ``example_inputs`` are ``Optional`` only because the
    orchestrator builds this from the previous stages' output, which is empty
    when those stages are skipped. **Both are required once the compilation
    stage executes**, and strategies should validate their presence.

    Attributes:
        soc_model: The target SoC (e.g., QcomChipset.SM8750). Required.
        backend_type: QNN backend type (HTP, GPU, LPAI, etc.). Required.
        model: The nn.Module to compile (quantized or original for FP16 mode).
            Required when the stage runs.
        example_inputs: Positional example inputs for ``torch.export``. Required
            when the stage runs. Sourced from the **model** via
            ``ModelLoaderAdapter.get_example_inputs``, never from calibration
            data: this tuple defines the exported graph's positional signature,
            supplies the zero-initialized KV caches a dataset sample does not
            carry, and fixes the AR length because HTP has no dynamic shapes.
        artifact_dir: Directory to store compiled artifacts.
        compile_specs: QNN compiler specifications for backend delegation.
    """

    soc_model: "QcomChipset"
    backend_type: "QnnExecuTorchBackendType"
    model: Optional["nn.Module"] = None
    example_inputs: Optional[Tuple[Any, ...]] = None
    artifact_dir: Path = field(default_factory=lambda: Path("."))
    compile_specs: Optional[List["CompileSpec"]] = None
