# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Registry-driven loading of a static LLM decoder.

The reference flow does construction, checkpoint acquisition and preparation in
one ~100-line ``LLMWrapper._prepare_model()`` with every per-model branch inlined
(``if decoder_model in {"gemma-2b", "gemma2-2b", "gemma3-1b"}: ...``). Here the
mechanism is written once and the per-model part is data:
:meth:`LLMLoaderAdapter.from_model_config` reads the model's row in
``models.model_registry``, which names the decoder class to construct and the
transforms to run.

All graph variants (calibration, decode, prefill) share a single checkpoint but
expose different ``get_example_inputs()`` signatures based on their ``ar_len`` and
KV-cache configuration. The adapter constructs all variants in :meth:`load_model`
and loads the same checkpoint into each graph wrapper.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from executorch.backends.qualcomm.genai_pipeline.strategies.model_preparation.default_model_loader_adapter import (
    DefaultModelLoaderAdapter,
)

logger = logging.getLogger(__name__)


class LLMLoaderAdapter(DefaultModelLoaderAdapter):
    """Loads a registry-declared static decoder from its checkpoint.

    Implementations of :class:`ModelLoaderAdapter` vary by *loading mechanism*,
    not by model family: this one covers every model in ``SUPPORTED_LLM_MODELS``,
    because what differs per model arrives as the registry row
    :meth:`from_model_config` reads rather than as a subclass.

    :meth:`load_model` returns a ``{ARTIFACT_TEXT_DECODER: {graph_name: module}}`` map.
    The graph variants share one checkpoint and differ only in AR length and
    KV-cache use.

    Args:
        model_config: The model's ``LLMModelConfig``, carrying ``repo_id``.
        control_args: The CLI argument namespace, providing the optional
            ``checkpoint`` override and consumed by ``TokenizerWrapper``.
    """

    RUNTIME_TOKENIZER_NAMES = ("tokenizer.json", "tokenizer.model")

    def __init__(
        self,
        model_config: Any,
        control_args: Any,
    ) -> None:
        self.model_config = model_config
        self.control_args = control_args

    def load_model(
        self,
        model_name: str,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the decoder's graph modules and load their weights.

        Constructs one module per graph variant (calibration, decode, prefill),
        each with its own ``ar_len`` and KV-cache configuration. All variants
        load the same checkpoint.

        Args:
            model_name: Registry model name (e.g. ``"llama3_2-1b_instruct"``).
            extra_options: Model preparation options containing component-keyed
                ``model_arch`` and ``weight_transforms`` entries, plus an
                optional component-keyed ``state_dict_loader`` map for HF
                checkpoints.

        Returns:
            ``{ARTIFACT_TEXT_DECODER: {graph_name: module}}``, all modules weight-carrying.
            A text-only model is the degenerate one-component case of the
            component-keyed shape the multimodal loader also produces.

        Raises:
            ValueError: If model_arch is missing or its text-decoder map is empty.
        """
        from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
            ARTIFACT_TEXT_DECODER,
        )

        extra_options = extra_options or {}

        model_arch = extra_options.get("model_arch")
        weight_transforms = (extra_options.get("weight_transforms") or {}).get(
            ARTIFACT_TEXT_DECODER, []
        )
        state_dict_loaders = extra_options.get("state_dict_loader") or {}
        if not isinstance(state_dict_loaders, dict):
            raise ValueError("state_dict_loader must be component-keyed")
        state_dict_loader = state_dict_loaders.get(ARTIFACT_TEXT_DECODER)

        if model_arch is None:
            raise ValueError("model_arch is required in model_options")

        graph_modules = self._build_custom_model(model_arch)
        state_dict = self._load_state_dict(state_dict_loader)
        state_dict = self._apply_weight_transforms(state_dict, weight_transforms)

        # For the nn.Module itself, different input-shape graphs share the same
        # module and weights. Today the wrapper constructors still produce one
        # module per graph shape, so every wrapper must receive the same weights.
        # TODO: Make module-wrapper get_example_inputs accept shape-related
        # parameters so different-shaped inputs are derived from the shape
        # request rather than from separate graph-wrapper modules.
        for module in graph_modules.values():
            module.load_state_dict(state_dict, strict=True, assign=True)

        logger.info(
            "Model loaded successfully with %d graph variants", len(graph_modules)
        )
        return {ARTIFACT_TEXT_DECODER: graph_modules}

    def _build_custom_model(
        self,
        model_arch: Any,
    ) -> Dict[str, Any]:
        """Construct the decoder modules, without weights.

        Args:
            model_arch: Component-keyed dict of per-graph constructors, i.e.
                ``{ARTIFACT_TEXT_DECODER: {graph_name: constructor}}``. Each constructor is
                a ``partial`` with its config and every arch-specific kwarg
                pre-bound by ``get_model_arch``, so this method stays arch-agnostic
                and calls each with no arguments.

        Returns:
            Dict mapping graph names to uninitialized modules.

        Raises:
            ValueError: If the text-decoder constructor map is empty.
        """
        from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
            ARTIFACT_TEXT_DECODER,
        )

        graph_arch = model_arch[ARTIFACT_TEXT_DECODER]
        if not graph_arch:
            raise ValueError("model_arch[ARTIFACT_TEXT_DECODER] must not be empty")

        modules: Dict[str, Any] = {}
        for name, build in graph_arch.items():
            modules[name] = build().eval()

        return modules

    def _load_state_dict(
        self,
        state_dict_loader: Optional[Callable[[str], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """The raw checkpoint state dict, before any weight transform.

        Whatever ``control_args`` names, or the model's ``repo_id`` downloaded
        and converted when it names none. Registry-specific HF loaders can be
        passed through ``model_options["state_dict_loader"]`` and must accept
        only ``repo_id``.

        Raises:
            ValueError: If the model ships no ``repo_id`` and ``control_args``
                names no checkpoint.
        """
        checkpoint = self.control_args.checkpoint
        if checkpoint is None:
            if self.model_config.repo_id is None:
                raise ValueError(
                    f"'{self.control_args.model}' is supplied as a local "
                    "checkpoint; pass one via control_args.checkpoint."
                )
            if state_dict_loader is not None:
                return state_dict_loader(self.model_config.repo_id)

            from executorch.examples.models.llama.hf_download import (
                download_and_convert_hf_checkpoint,
            )

            # convert_weights is held as a class attribute, so reading it off the
            # model_config instance binds it; __func__ recovers the plain function
            # the downloader expects.
            checkpoint = download_and_convert_hf_checkpoint(
                self.model_config.repo_id, self.model_config.convert_weights.__func__
            )

        return torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)

    def _apply_weight_transforms(
        self,
        state_dict: Dict[str, Any],
        weight_transforms: Optional[List[Callable]] = None,
    ) -> Dict[str, Any]:
        """Apply weight transforms to the state dict.

        Args:
            state_dict: The raw state dict to transform.
            weight_transforms: Optional list of transforms to apply in order.

        Returns:
            The transformed state dict.
        """
        if not weight_transforms:
            return state_dict

        logger.debug("Applying %d weight transforms", len(weight_transforms))
        for transform in weight_transforms:
            state_dict = transform(state_dict)

        return state_dict

    def get_inference(
        self,
        meta: Dict[str, Any],
        example_inputs: Dict[str, Any],
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Build the ``ModelInference`` bound to the calibration graph.

        The calibration graph is the non-deployed ``GRAPH_FORWARD`` graph;
        this selects it from the full per-graph maps and drives a
        ``DecoderInference`` from its metadata and export-input signature.

        Args:
            meta: ``{ARTIFACT_TEXT_DECODER: {graph_name: meta}}`` from :meth:`get_metadata`.
            example_inputs: ``{ARTIFACT_TEXT_DECODER: {graph_name: inputs}}`` from
                :meth:`get_example_inputs`.
            extra_options: Reads ``embedding_quantize`` to decide the token dtype.

        Returns:
            A ``ModelInference`` wrapping the calibration graph's
            ``DecoderInference``, or ``None`` when no calibration graph is present.
        """
        from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
            ARTIFACT_TEXT_DECODER,
        )
        from executorch.backends.qualcomm.genai_pipeline.graph_names import (
            GRAPH_FORWARD,
        )
        from executorch.examples.qualcomm.oss_scripts.llama.inference import (
            DecoderInference,
            ModelInference,
        )

        calibration_meta = meta.get(ARTIFACT_TEXT_DECODER, {}).get(GRAPH_FORWARD)
        calibration_inputs = example_inputs.get(ARTIFACT_TEXT_DECODER, {}).get(
            GRAPH_FORWARD
        )
        if calibration_meta is None or calibration_inputs is None:
            return None

        extra_options = extra_options or {}
        use_i64_token = extra_options.get("embedding_quantize") is not None

        return ModelInference(
            decoder=DecoderInference(
                get_example_inputs=lambda: calibration_inputs,
                max_context_len=calibration_meta["get_max_context_len"],
                max_batch_size=calibration_meta["get_max_batch_size"],
                use_i64_token=use_i64_token,
            )
        )

    def load_tokenizer(
        self,
        model_name: str,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Build the ``TokenizerWrapper`` for this model.

        Returns the wrapper, not the bare tokenizer: the dataset builders and
        evaluators need its chat-template and multimodal-prompt helpers, and it
        also resolves ``runtime_tokenizer_path``. The bare tokenizer remains
        reachable as ``.tokenizer``.
        """
        from executorch.examples.qualcomm.oss_scripts.llama.tokenizer import (
            TokenizerWrapper,
        )

        # TODO: Remove this compatibility bridge once TokenizerWrapper accepts
        # the pipeline's ``model`` and ``artifact_dir`` field names directly.
        self.control_args.decoder_model = getattr(
            self.control_args, "decoder_model", self.control_args.model
        )
        self.control_args.artifact = getattr(
            self.control_args, "artifact", self.control_args.artifact_dir
        )

        return TokenizerWrapper(self.control_args, self.model_config)

    def get_example_inputs(
        self,
        model: Any,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build example inputs for ``torch.export`` from the model itself.

        Prefers the model's own ``get_example_inputs()`` when it exposes one, so
        models that already describe their export signature (the LLM wrappers
        build a flat ``(tokens, attn_mask, pos_ids, *k_caches, *v_caches)``
        tuple) stay authoritative. Otherwise a minimal ``(input_ids,)`` is
        synthesized, which is the correct signature for a plain HuggingFace
        causal LM without an external KV cache.

        Args:
            model: The module returned by :meth:`load_model`.

        Returns:
            A flat tuple positionally matching ``model.forward``.
        """

        extra_options = extra_options or {}

        model_provided = getattr(model, "get_example_inputs", None)
        if callable(model_provided):
            logger.info("Using example inputs provided by the model")
            return tuple(model_provided())

        raise ValueError(
            "Module has no get_example_inputs() method and "
            "no example_inputs provided in extra_options"
        )

    def export_tokenizer(
        self,
        tokenizer: Any,
        output_dir: Path,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Export tokenizer to disk and return the runtime tokenizer file.

        ``TokenizerWrapper`` has already written the artifacts. Prefer the
        runtime formats recognized by ``pytorch_tokenizers.get_tokenizer`` over
        the wrapper's order-dependent fallback.

        Args:
            tokenizer: The tokenizer instance to export.
            output_dir: Directory to write the exported tokenizer artifacts to.
            extra_options: Additional export options.

        Returns:
            Path to the runtime tokenizer file (e.g. ``tokenizer.json``).

        Raises:
            FileNotFoundError: If no tokenizer artifacts were written.
        """
        runtime_tokenizer_path = Path(tokenizer.runtime_tokenizer_path)
        artifact_dir = Path(tokenizer.artifact)
        artifacts = list(artifact_dir.iterdir()) if artifact_dir.is_dir() else []
        if runtime_tokenizer_path not in artifacts:
            artifacts.append(runtime_tokenizer_path)
        return self._select_runtime_tokenizer(artifacts)

    @classmethod
    def from_model_config(
        cls,
        model_config: Any,
        *,
        control_args: Any,
    ) -> "LLMLoaderAdapter":
        """Build the adapter, taking its source transforms as pre-bound partials.

        Which transforms this model uses is resolved by
        ``model_lookup.get_source_transform`` and passed in here, so the adapter
        neither reads a registry nor knows the per-model mapping.
        """
        return cls(
            model_config=model_config,
            control_args=control_args,
        )
