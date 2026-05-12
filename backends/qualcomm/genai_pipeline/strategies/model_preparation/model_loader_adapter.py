# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class ModelLoaderAdapter(Protocol):
    """Protocol for model and tokenizer loading operations.

    Wraps external model loading APIs (HuggingFace ``AutoModelForCausalLM``,
    ``AutoTokenizer``, etc.) behind an injectable interface for testability.

    Scope is limited to acquiring the model and its tokenizer. Datasets are a
    cross-stage concern and live in ``genai_pipeline.datasets``
    (:class:`CalibrationDataAdapter`, :class:`TrainingDataAdapter`).

    .. note::
        Implementations vary by **loading mechanism**, not by model family.
        ``DefaultModelLoaderAdapter`` covers every text-only causal LM reachable
        via HuggingFace ``AutoModelForCausalLM``; a new implementation is only
        warranted when the mechanism itself differs (multimodal models needing
        ``AutoModel`` / ``AutoModelForSpeechSeq2Seq``, a GGUF loader, a local
        checkpoint format, ...).

        Per-model **graph and weight transformations** are deliberately *not*
        expressed by subclassing this protocol -- overlapping transform sets
        (e.g. Llama needing ``[A, B, C]`` while Gemma needs ``[B, C, D]``) would
        be reimplemented in each subclass. They are instead declared as data on
        the model registry entry, so each transform is implemented once and
        shared, and adding a model is a registry row rather than a new class.

        The exact shape of that declaration is intentionally left open here: the
        transforms in the existing flow are not uniform -- some rewrite the
        state dict before it is loaded, some need the constructed module, and
        some replace the module -- and the ordering between those kinds matters.
        Pinning a single flat transform list now would encode the wrong
        contract, so the registry columns land with the transforms themselves
        once each is extracted into a named, shared function.

    A single call returns **one** model. Multi-graph export configurations of the
    same weights (hybrid prefill/decode, a separate token-embedding graph) are
    expanded by the quantization and compilation *strategies*, and genuinely
    multi-module models (multimodal encoders) belong in a specialized adapter.
    """

    def load_model(
        self,
        model_name: str,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Load a model by name or path.

        Args:
            model_name: Model identifier (e.g., HuggingFace model ID or local path).
            extra_options: Additional model loading options (dtype, device_map, etc.).

        Returns:
            The loaded nn.Module.
        """
        ...

    def load_tokenizer(
        self,
        model_name: str,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Load the tokenizer for the given model.

        Args:
            model_name: Model identifier matching the model.
            extra_options: Additional tokenizer options.

        Returns:
            The tokenizer instance.
        """
        ...

    def export_tokenizer(
        self,
        tokenizer: Any,
        output_dir: Path,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Export tokenizer for on-device runtime use.

        Args:
            tokenizer: The tokenizer instance to export.
            output_dir: Directory to write the exported tokenizer artifacts to.
            extra_options: Additional export options.

        Returns:
            Path to the exported runtime tokenizer **file** (not the containing
            directory). The returned path must be directly loadable by
            ``pytorch_tokenizers.get_tokenizer`` and the C++
            ``llm::load_tokenizer`` runtime, both of which expect a single file
            such as ``tokenizer.json`` or ``tokenizer.model``.
        """
        ...
