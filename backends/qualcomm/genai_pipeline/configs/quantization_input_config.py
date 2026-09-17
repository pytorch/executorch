# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from executorch.backends.qualcomm.serialization.qc_schema import (
        QcomChipset,
        QnnExecuTorchBackendType,
    )
    from torch import nn


@dataclass
class QuantizationInputConfig:
    """Input configuration for the quantization stage.

    ``model_module``, ``example_inputs`` and ``calibration_data`` are
    ``Optional`` only because the orchestrator builds this from the previous
    stage's output, which is empty when model preparation is skipped. **All
    three are required once the quantization stage executes**, and strategies
    should validate their presence.

    Flows needing no quantization (FP16, GPU backends) skip the stage entirely
    via ``GenAIPipeline.from_proxy(proxy, skip_stages={STAGE_QUANTIZATION})``
    rather than entering it with a no-op strategy; the orchestrator then returns
    an empty ``QuantizationOutputConfig()`` and compilation receives the
    unquantized module.

    Attributes:
        soc_model: The target SoC (e.g., QcomChipset.SM8750). Required.
        backend_type: QNN backend type (HTP, GPU, LPAI, etc.). Required.
        model_module: The nn.Module to quantize. Required when the stage runs.
        example_inputs: Positional example inputs for ``torch.export``. Required
            when the stage runs. Sourced from the **model** via
            ``ModelLoaderAdapter.get_example_inputs``, never from
            ``calibration_data``: this tuple defines the exported graph's
            positional signature, supplies the zero-initialized KV caches a
            dataset sample does not carry, and fixes the AR length because HTP
            has no dynamic shapes.
        calibration_data: Calibration samples. Required when the stage runs. Any
            Iterable[Tuple[Tensor, ...]], including a DataLoader. Consumed only
            by ``calibrate()`` -- it is never indexed or peeked at, so a
            single-use generator stays intact.
        training_data: Training dataset for quantization-aware training (QAT),
            typically (features, labels) pairs. Mirrors ``qat_training_data`` in
            ``build_executorch_binary``. ``None`` selects PTQ.
        quant_recipe: Quantization recipe (per-layer bit widths, group sizes, etc.).
        extra_options: Additional quantization-specific options.
    """

    soc_model: "QcomChipset"
    backend_type: "QnnExecuTorchBackendType"
    model_module: Optional["nn.Module"] = None
    example_inputs: Optional[Tuple[Any, ...]] = None
    calibration_data: Optional[Iterable[Any]] = None
    training_data: Optional[Iterable[Any]] = None
    quant_recipe: Any = None
    extra_options: Dict[str, Any] = field(default_factory=dict)
