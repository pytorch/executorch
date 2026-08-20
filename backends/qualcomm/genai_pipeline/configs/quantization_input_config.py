# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from executorch.backends.qualcomm.serialization.qc_schema import (
        QcomChipset,
        QnnExecuTorchBackendType,
    )
    from torch import nn


@dataclass
class QuantizationInputConfig:
    """Input configuration for the quantization stage.

    Attributes:
        soc_model: The target SoC (e.g., QcomChipset.SM8750). Required.
        backend_type: QNN backend type (HTP, GPU, LPAI, etc.). Required.
        model_module: The nn.Module to quantize.
        calibration_data: Calibration dataset samples.
        quant_recipe: Quantization recipe (per-layer bit widths, group sizes, etc.).
        extra_options: Additional quantization-specific options.
    """

    soc_model: "QcomChipset"
    backend_type: "QnnExecuTorchBackendType"
    model_module: Optional["nn.Module"] = None
    calibration_data: Optional[List[Any]] = None
    quant_recipe: Any = None
    extra_options: Dict[str, Any] = field(default_factory=dict)
