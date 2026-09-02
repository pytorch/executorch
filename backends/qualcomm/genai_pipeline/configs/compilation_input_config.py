# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

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
    ``graphs`` describes a model's components, each with its own graphs, which
    is what a decoder needs: its AR-N prefill and AR-1 decode graphs come from
    the same weights and are lowered together into one multi-method ``.pte`` so
    they can share them. The single-graph form is the degenerate case of the
    second, kept because it is what a non-decoder model needs.

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
        graphs: The graphs to compile, keyed by artifact key (see
            ``artifact_keys``) and then by graph name (see ``graph_names``).
            The outer level is one ``.pte`` per key; the inner is one method
            within it, so a text-only model is the degenerate
            ``{ARTIFACT_TEXT_DECODER: {...}}``. Produced by the quantization
            stage, which reconciles the encodings and releases the calibration
            graph first, so every bundle here is deployable. ``None`` on the
            single-graph path, where ``model`` and ``example_inputs`` are used
            instead.
        num_sharding: How many shards a component's graphs are split into,
            either per component or as one count describing the decoder.

            Sharding is per-component: the decoder's count comes from the model
            config while an encoder's comes from its own
            (``GraniteSpeechEncoder.num_sharding``), so one scalar cannot
            describe a multimodal model. A ``Dict[artifact_key, int]`` is the
            general form; a plain ``int`` is accepted as shorthand for "the
            decoder's count", which is what a text-only flow has.

            Drives both the multiple-contexts compile spec and the insertion of
            the splitting pass, and deliberately from the same field: a model
            sharded by the specs but never split by the passes compiles cleanly
            and is wrong. The authoritative source is the model registry, which
            reaches here via ``ModelPreparationOutputConfig.num_shardings``.
        meta: Graph metadata, keyed by component and graph name. Used to build
            unquantized bundles from ``model`` when ``graphs`` is empty, which
            happens when the quantization stage is skipped (FP16 or GPU).
    """

    soc_model: "QcomChipset"
    backend_type: "QnnExecuTorchBackendType"
    model: Optional["nn.Module"] = None
    example_inputs: Optional[Tuple[Any, ...]] = None
    artifact_dir: Path = field(default_factory=lambda: Path("."))
    compile_specs: Optional[List["CompileSpec"]] = None
    graphs: Optional[Dict[str, Dict[str, "GraphBundle"]]] = None
    num_sharding: Union[int, Dict[str, int]] = 1
    meta: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None
