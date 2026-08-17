# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Protocol,
    runtime_checkable,
    Sequence,
    Tuple,
)


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

    ``load_model`` returns the full graph map:
    ``{component: {graph_name: module}}``. Text-only models are the one-component
    ``{ARTIFACT_TEXT_DECODER: {...}}`` case; multimodal adapters add encoder and
    token-embedding components. Example inputs and metadata retain this shape
    because they vary by graph.

    The strategy drops deployed graph modules and flattens the selected weight
    holders to ``{component: module}``, for example ``{ARTIFACT_TEXT_DECODER: decoder}``.
    The graph axis is absent because the selected module is shared when
    exporting every graph variant for that component; only its inputs and
    metadata differ.
    """

    def load_model(
        self,
        model_name: str,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Load the model, weights and all.

        The whole weight stage lives here: acquiring the checkpoint, rewriting it
        (key renames, value transforms) and loading it into the module. The
        module this returns therefore already carries its final weights;
        :meth:`apply_module_transforms` then handles the module-level preparation.

        Args:
            model_name: Model identifier (e.g., HuggingFace model ID or local path).
            extra_options: Additional model loading options (dtype, device_map, etc.).

        Returns:
            The weight-carrying ``{component: {graph_name: module}}`` map.
        """
        ...

    def apply_module_transforms(
        self,
        module: Any,
        module_transforms: Optional[Sequence[Callable[[Any], Any]]] = None,
    ) -> Any:
        """Apply transforms to one component module.

        The stage that owns the modules calls this, not :meth:`load_model`: a
        module transform may *replace* a module (``convert_linear_to_conv2d``
        returns a new one) rather than mutate it, so the caller must own the
        result. Implementations whose models need no preparation return the
        modules unchanged.

        The weight stage is not here -- it runs inside :meth:`load_model`, the
        only holder of the checkpoint -- so no caller ever sees a state dict.

        Args:
            module: One value from the single-level ``{component: module}`` map
                selected by the strategy.
            module_transforms: Transforms to apply, in order.

        Returns:
            The transformed module. Callers must use the return value.
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

    def get_example_inputs(
        self,
        model: Any,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, ...]:
        """Build the positional example inputs for ``torch.export``.

        These come from the **model**, never from the calibration dataset: the
        tuple defines the exported graph's positional signature, including the
        zero-initialized KV cache entries a dataset sample does not carry, and
        it bakes in the AR length (HTP has no dynamic shapes). The dependency
        runs model -> dataset, not the reverse: the reference flow derives the
        dataset's attention-mask schema *from* the example input
        (``LLMWrapper.attn_mask`` returns ``example_input[1]``).

        Args:
            model: One graph module from ``load_model``'s nested map.
            extra_options: Additional options controlling the example shapes.

        Returns:
            A flat tuple positionally matching ``model.forward``, ready to pass
            straight to ``torch.export.export(model, example_inputs)``.
        """
        ...

    def get_metadata(
        self,
        module: Any,
    ) -> Any:
        """Read one graph module's constant metadata for the ``.pte``.

        The metadata (``get_n_layers``, ``get_head_dim``, ``get_max_context_len``,
        ...) is baked into the compiled binary as constant methods and is also
        needed downstream to reconstruct logits / KV-cache shapes during
        quantization encoding reconciliation.

        Args:
            module: One graph module from the component/graph map returned by
                :meth:`load_model`.

        Returns:
            The graph metadata, or an empty value when the module exposes none.
            The strategy owns routing this into ``{component: {graph_name: meta}}``.
        """
        ...

    def get_inference(
        self,
        meta: Any,
        example_inputs: Any,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Build the ``ModelInference`` that drives PTQ calibration.

        The inference object is bound to the calibration graph -- the
        non-deployed, KV-cache graph whose observers calibration populates. The
        strategy passes the full per-graph metadata and example inputs; the loader
        selects its calibration graph and assembles the ``ModelInference``, since
        the inference shape (decoder-only vs decoder + encoder) is model-family
        specific.

        Args:
            meta: The full per-graph metadata from :meth:`get_metadata`
                (``{component: {graph_name: meta}}`` or ``{graph_name: meta}``).
            example_inputs: The full per-graph example inputs from
                :meth:`get_example_inputs`, mirroring ``meta``'s shape.
            extra_options: Additional options; reads ``embedding_quantize`` to
                decide the token dtype.

        Returns:
            A ``ModelInference`` wrapping a ``DecoderInference`` for the
            calibration graph, or ``None`` when the model has no calibration
            driver.
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
