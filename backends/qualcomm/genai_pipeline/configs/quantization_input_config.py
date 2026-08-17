# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from executorch.backends.qualcomm.serialization.qc_schema import (
        QcomChipset,
        QnnExecuTorchBackendType,
    )
    from torch import nn


@dataclass
class QuantizationInputConfig:
    """Input configuration for the quantization stage.

    ``model_module`` and ``example_inputs`` are ``Optional`` only because the
    orchestrator builds this from the previous stage's output, which is empty
    when model preparation is skipped. **Both are required once the
    quantization stage executes**, and strategies should validate their
    presence.

    Flows needing no quantization (FP16, GPU backends) skip the stage entirely
    via ``GenAIPipeline.from_proxy(proxy, skip_stages={STAGE_QUANTIZATION})``
    rather than entering it with a no-op strategy; the orchestrator then returns
    an empty ``QuantizationOutputConfig()`` and compilation receives the
    unquantized module.

    ``model_module`` is keyed by component. ``example_inputs`` and ``meta`` add
    the inner graph axis, because their values vary by graph.

    Attributes:
        soc_model: The target SoC (e.g., QcomChipset.SM8750). Required.
        backend_type: One QNN backend type (HTP, GPU, LPAI, etc.) shared by all
            components. Per-component backend routing is unsupported.
        model_module: Prepared modules, keyed by component. Required when the
            stage runs.
        example_inputs: Positional example inputs for ``torch.export``, per
            component and graph. Required when the stage runs. Sourced from the **model** via
            ``ModelLoaderAdapter.get_example_inputs``, never from a dataset
            sample: this tuple defines the exported graph's positional
            signature, supplies the zero-initialized KV caches a dataset sample
            does not carry, and fixes the AR length because HTP has no dynamic
            shapes.
        tokenizer: The TokenizerWrapper from model preparation, used by the
            strategy to build calibration data.
        training_data: Training dataset for quantization-aware training (QAT),
            typically (features, labels) pairs. Mirrors ``qat_training_data`` in
            ``build_executorch_binary``. ``None`` selects PTQ.
        quant_recipe: Quantization recipe, or a per-component map of them.
        meta: Per-graph constant metadata from model preparation.
        inference: Optional model-specific inference instance or callable used
            by PTQ calibration. Created during model preparation; ``None`` when
            the model does not provide one.
        extra_options: Additional quantization-specific options.
    """

    soc_model: "QcomChipset"
    backend_type: "QnnExecuTorchBackendType"
    model_module: Optional[Dict[str, "nn.Module"]] = None
    example_inputs: Optional[Dict[str, Dict[str, Tuple[Any, ...]]]] = None
    tokenizer: Any = None
    quant_recipe: Any = None
    meta: Optional[Dict[str, Any]] = None
    inference: Optional[Any] = None
    extra_options: Dict[str, Any] = field(default_factory=dict)
