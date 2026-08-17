# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn


@dataclass
class ModelPreparationOutputConfig:
    """Output produced by the model preparation stage.

    All fields are ``Optional`` because ``GenAIPipeline._run_model_preparation``
    returns an empty instance when the stage is skipped (e.g. compile-only flows
    with a caller-supplied module). ``None`` is a "stage did not run" sentinel,
    not a valid post-execution state: when the stage runs, ``model_module`` and
    ``tokenizer`` are always populated.

    ``model_module`` is keyed by component. ``example_inputs`` and ``meta`` add
    the inner graph axis, because their values vary by graph.

    Attributes:
        model_module: The prepared modules, keyed by component.
        tokenizer: The tokenizer instance for encoding/decoding text.
        example_inputs: Per-component, per-graph positional example inputs for
            ``torch.export`` (``{component: {graph: tuple}}``), derived from the
            **model** (never from the calibration data): they carry each exported
            graph's signature, its zero-initialized KV caches, and the AR length
            baked in because HTP has no dynamic shapes. The dependency runs
            model -> dataset, not the reverse -- the calibration data's
            attention-mask schema is itself derived from these tuples.
        runtime_tokenizer_path: Path to the runtime tokenizer **file** (not the
            containing directory) for on-device inference.
        chat_template: Optional chat template for instruct models.
        meta: Per-graph ``get_metadata()`` constants (layer count, head dim,
            context/AR lengths). Feeds logits / KV-cache shape reconstruction in
            quantization and is baked into the ``.pte``.
        inference: The ``ModelInference`` bound to the calibration graph, built
            by the loader adapter. Drives PTQ calibration in the quantization
            stage; ``None`` for flows that build no inference (e.g. multimodal,
            not yet supported).
        num_shardings: Optional per-component number of shardings for the model.
    """

    model_module: Optional[Dict[str, "nn.Module"]] = None
    tokenizer: Any = None
    example_inputs: Optional[Dict[str, Dict[str, Tuple[Any, ...]]]] = None
    runtime_tokenizer_path: Optional[Path] = None
    chat_template: Optional[str] = None
    meta: Optional[Dict[str, Dict[str, Any]]] = None
    inference: Any = None
    num_shardings: Optional[Dict[str, int]] = None
