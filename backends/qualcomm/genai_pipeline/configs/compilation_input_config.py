# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from executorch.backends.qualcomm.genai_pipeline.graph_bundle import GraphBundle
    from executorch.backends.qualcomm.serialization.qc_schema import (
        QcomChipset,
        QnnExecuTorchBackendType,
    )
    from executorch.exir.backend.compile_spec_schema import CompileSpec
    from torch import nn


@dataclass
class CompilationInputConfig:
    """Input configuration for the compilation stage.

    ``model`` and ``example_inputs`` are ``Optional`` only because the
    orchestrator builds this from the previous stages' output, which is empty
    when those stages are skipped. **Both are required once the compilation
    stage executes**, and strategies should validate their presence.

    There are two ways to name what to compile, and a strategy takes whichever
    is populated. ``model`` plus ``example_inputs`` describes a single graph.
    ``graphs`` describes several exported from the same weights -- a hybrid
    decoder's AR-N prefill and AR-1 decode -- which are lowered together into
    one multi-method ``.pte`` so they can share weights. The single-graph form
    is the degenerate case of the second, kept because it is what a
    non-decoder model needs.

    Attributes:
        soc_model: The target SoC (e.g., QcomChipset.SM8750). Required.
        backend_type: QNN backend type (HTP, GPU, LPAI, etc.). Required.
        model: The nn.Module to compile (quantized or original for FP16 mode).
            Required when the stage runs.
        example_inputs: Positional example inputs for ``torch.export``. Belongs
            to the single-graph form only, alongside ``model``: it is required
            when the stage runs on that path and ``None`` when ``graphs`` is
            populated, where each ``GraphBundle`` carries its own ``inputs``.
            Sourced from the **model** via
            ``ModelLoaderAdapter.get_example_inputs``, never from calibration
            data: this tuple defines the exported graph's positional signature,
            supplies the zero-initialized KV caches a dataset sample does not
            carry, and fixes the AR length because HTP has no dynamic shapes.
        artifact_dir: Directory to store compiled artifacts.
        compile_specs: QNN compiler specifications for backend delegation.
        graphs: The graphs to compile, keyed by graph name (see
            ``graph_names.DECODER_GRAPH_NAMES``), each carrying its own
            module, inputs and metadata. Produced by the quantization stage,
            which reconciles their encodings and releases the calibration graph
            first, so every bundle here is deployable. ``None`` on the
            single-graph path, where ``model`` and ``example_inputs`` are used
            instead.
    """

    soc_model: "QcomChipset"
    backend_type: "QnnExecuTorchBackendType"
    model: Optional["nn.Module"] = None
    example_inputs: Optional[Tuple[Any, ...]] = None
    artifact_dir: Path = field(default_factory=lambda: Path("."))
    compile_specs: Optional[List["CompileSpec"]] = None
    graphs: Optional[Dict[str, "GraphBundle"]] = None
