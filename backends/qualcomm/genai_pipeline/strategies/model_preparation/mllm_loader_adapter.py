# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model loading for a multimodal model, every component implemented here.

A multimodal model is several modules -- one or more encoders, a token
embedding, and a text decoder. Loading them has no inter-dependency (unlike
quantization's ordered chain), so the whole multi-module load lives in this one
adapter.

This is the multimodal implementation of ``ModelLoaderAdapter``. It uses the
same component/graph map contract as the text-only loader, with extra
components for modality encoders and token embedding.

It implements every component itself rather than delegating the decoder to
``LLMLoaderAdapter``:

- **text_decoder**: constructed per graph from model_arch, its
  checkpoint acquired and rewritten through ``get_source_transform``'s partials,
  then loaded strict/assign -- the same mechanism the single-module adapter uses,
  written here so this adapter owns its whole load.
- **encoder(s)** and **tok_embedding**: both derived from a single shared
  HuggingFace ``auto_model`` -- the encoder through the modality config's
  ``create_encoder``, the token embedding from ``auto_model.get_input_embeddings()``
  sized by the decoder's ``model_args``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_AUDIO_ENCODER,
    ARTIFACT_TEXT_DECODER,
    ARTIFACT_TOK_EMBEDDING,
    ARTIFACT_VISION_ENCODER,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.model_preparation.default_model_loader_adapter import (
    DefaultModelLoaderAdapter,
)

logger = logging.getLogger(__name__)


class MLLMLoaderAdapter(DefaultModelLoaderAdapter):
    """Loads every component of a multimodal model.

    Orchestrates loading of encoders, token embedding, and text decoder from a
    shared HuggingFace model. Each component is loaded independently with no
    inter-dependencies, enabling flexible composition of multimodal architectures.

    The decoder and embedding are constructed with multiple graph variants
    (calibration, decode, prefill) to support different inference modes. All
    variants are derived from a single nn.Module via torch.export.export() with
    different input shapes (ar_len, KV-cache config), so only one module per
    component needs to hold the checkpoint weights.

    Args:
        model_config: The model's ``LLMModelConfig``, carrying ``repo_id``, the
            decoder class (``model_arch``) and per-modality attributes
            (``vision_encoder`` / ``audio_encoder``).
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

        self._modalities = tuple(
            m
            for m in (ARTIFACT_VISION_ENCODER, ARTIFACT_AUDIO_ENCODER)
            if hasattr(model_config, m)
        )
        if not self._modalities:
            raise ValueError(
                "MLLMLoaderAdapter requires at least one modality encoder "
                f"({ARTIFACT_VISION_ENCODER!r} or {ARTIFACT_AUDIO_ENCODER!r}) to be defined "
                f"in {type(model_config).__name__}."
            )

    def load_model(
        self,
        model_name: str,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Build every component's graph modules.

        Orchestrates loading of all model components (encoders, token embedding,
        text decoder) from a shared HuggingFace model. Each component is loaded
        independently with no inter-dependencies.

        The decoder and embedding are constructed with multiple graph variants
        (calibration, decode, prefill) to support different inference modes.

        Args:
            model_name: Registry model name (e.g. ``"smolvlm_500m"``). Unused;
                the model is identified by ``control_args``.
            extra_options: Model preparation options containing component-keyed
                ``model_arch`` and ``weight_transforms`` entries, plus an
                optional component-keyed ``state_dict_loader`` map for HF
                checkpoints.

        Returns:
            ``{component: {graph_name: module}}``, where component is one of
            ARTIFACT_TEXT_DECODER, ARTIFACT_TOK_EMBEDDING, ARTIFACT_VISION_ENCODER, or ARTIFACT_AUDIO_ENCODER.

        Raises:
            ValueError: If required model_options are missing or invalid.
        """
        extra_options = extra_options or {}

        # MLLM model options are component-keyed so each component can own its
        # graph constructors and transforms.
        decoder_model_arch = extra_options.get("model_arch")
        if not isinstance(decoder_model_arch, dict):
            raise ValueError(
                f"For multimodal models, model_arch must be a component-keyed dict "
                f"(e.g., {{ARTIFACT_TEXT_DECODER: value}}), got {type(decoder_model_arch).__name__}"
            )
        embedding_model_arch = decoder_model_arch.get(ARTIFACT_TOK_EMBEDDING)
        decoder_model_arch = decoder_model_arch.get(ARTIFACT_TEXT_DECODER)
        if decoder_model_arch is None:
            raise ValueError(
                "model_arch dict must contain ARTIFACT_TEXT_DECODER key for multimodal models"
            )

        decoder_weight_transforms = extra_options.get("weight_transforms") or {}
        if not isinstance(decoder_weight_transforms, dict):
            raise ValueError(
                f"For multimodal models, weight_transforms must be a component-keyed dict "
                f"(e.g., {{ARTIFACT_TEXT_DECODER: value}}), got {type(decoder_weight_transforms).__name__}"
            )
        decoder_weight_transforms = decoder_weight_transforms.get(
            ARTIFACT_TEXT_DECODER, []
        )
        if decoder_weight_transforms is None:
            decoder_weight_transforms = []
        state_dict_loaders = extra_options.get("state_dict_loader") or {}
        if not isinstance(state_dict_loaders, dict):
            raise ValueError("state_dict_loader must be component-keyed")
        state_dict_loader = state_dict_loaders.get(ARTIFACT_TEXT_DECODER)

        # Load auto model
        auto_model = self._load_auto_model()

        modules: Dict[str, Dict[str, Any]] = {
            # Load encoder
            **{
                modality: self._load_encoder(modality, auto_model)
                for modality in self._modalities
            },
            # Load token embedding (constructors carry their own shape kwargs)
            ARTIFACT_TOK_EMBEDDING: self._load_embedding(
                auto_model,
                embedding_model_arch,
            ),
            # Load text decoder
            ARTIFACT_TEXT_DECODER: self._load_decoder(
                decoder_model_arch,
                decoder_weight_transforms,
                state_dict_loader,
            ),
        }

        return modules

    def _load_encoder(self, modality: str, auto_model: Any) -> Dict[str, Any]:
        """Load one modality encoder from the shared HuggingFace model.

        Creates an encoder wrapper for the specified modality (vision or audio)
        using the modality config's ``create_encoder`` method. The encoder weights
        are loaded from the shared HuggingFace model with strict=False to allow
        the wrapper to expose only the relevant modality weights.

        Args:
            modality: The modality name (ARTIFACT_VISION_ENCODER or ARTIFACT_AUDIO_ENCODER).
            auto_model: The shared HuggingFace model.

        Returns:
            ``{GRAPH_FORWARD: encoder_module}``.

        Raises:
            AttributeError: If the modality config is not found in model_config.
        """
        from executorch.backends.qualcomm.genai_pipeline.graph_names import (
            GRAPH_FORWARD,
        )

        logger.info("Loading %s", modality)
        modality_config = getattr(self.model_config, modality)
        encoder = modality_config().create_encoder(auto_model.config).eval()
        # strict=False: the encoder wrapper exposes only this modality's weights.
        encoder.load_state_dict(auto_model.state_dict(), strict=False)
        return {GRAPH_FORWARD: encoder}

    def _load_embedding(
        self,
        auto_model: Any,
        embedding_arch: Any,
    ) -> Dict[str, Any]:
        """Load the token embedding for all its graph variants.

        Creates one embedding module per graph variant (calibration, decode,
        prefill), keyed by the token-embedding graph names carried in
        ``embedding_arch``. All variants share the same embedding weights
        from the HuggingFace model.

        Args:
            auto_model: The shared HuggingFace model.
            embedding_arch: ``{graph_name: constructor}`` for the token embedding.
                Each constructor is a ``partial`` with the embedding's shape kwargs
                pre-bound; the shared weights are the sole call-time argument.

        Returns:
            ``{graph_name: embedding_module}``, all modules weight-carrying.
        """
        import torch

        if not embedding_arch:
            raise RuntimeError(
                "embedding_arch not available. Ensure load_model() is "
                "called with model_options containing the token-embedding arch."
            )

        logger.info("Loading token embedding")
        embedding_weights = auto_model.get_input_embeddings().to(torch.float32)

        embedding_modules: Dict[str, Any] = {}

        # Each constructor is a partial with the embedding's shape kwargs pre-bound;
        # the shared HuggingFace weights are the sole
        # call-time argument.
        for graph_name, build in embedding_arch.items():
            embedding_modules[graph_name] = build(embedding_weights)

        logger.info(
            "Token embedding loaded successfully with %d graph variants",
            len(embedding_modules),
        )
        return embedding_modules

    def _load_decoder(
        self,
        model_arch: Any = None,
        weight_transforms: Optional[List[Callable]] = None,
        state_dict_loader: Optional[Callable[[str], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Load the text decoder component's modules.

        Constructs one decoder module per graph variant (calibration, decode, prefill),
        each with its own ``ar_len`` and KV-cache configuration. All variants
        load the same checkpoint.

        Args:
            model_arch: ``{graph_name: constructor}`` for the decoder. Each
                constructor is a ``partial`` with its config and construction kwargs
                pre-bound by ``get_model_arch``, so it is called with no arguments.
            weight_transforms: Optional list of transforms to apply to the state dict
                in order before loading.
            state_dict_loader: Optional HF checkpoint loader that receives only
                ``repo_id`` and returns a state dict.

        Returns:
            Dict mapping graph names to decoder modules, all weight-carrying.

        Raises:
            ValueError: If model_arch is missing or empty.
        """
        if not model_arch:
            raise ValueError("model_arch is required and must not be empty")

        logger.info("Loading text decoder")
        decoder_modules = self._build_custom_decoder_model(model_arch)
        state_dict = self._load_decoder_state_dict(state_dict_loader)

        if weight_transforms:
            for transform in weight_transforms:
                state_dict = transform(state_dict)

        # For the nn.Module itself, different input-shape graphs share the same
        # module and weights. Today the wrapper constructors still produce one
        # module per graph shape, so every wrapper must receive the same weights.
        # TODO: Make module-wrapper get_example_inputs accept shape-related
        # parameters so different-shaped inputs are derived from the shape
        # request rather than from separate graph-wrapper modules.
        for module in decoder_modules.values():
            module.load_state_dict(state_dict, strict=True, assign=True)

        logger.info(
            "Text decoder loaded successfully with %d graph variants",
            len(decoder_modules),
        )

        return decoder_modules

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

        model_provided = getattr(model, "get_example_inputs", None) or getattr(
            model, "get_example_input", None
        )
        if callable(model_provided):
            logger.info("Using example inputs provided by the model")
            return tuple(model_provided())

        raise ValueError(
            "Module has no get_example_inputs() method and "
            "no example_inputs provided in extra_options"
        )

    def get_inference(
        self,
        meta: Dict[str, Any],
        example_inputs: Dict[str, Any],
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Build the ``ModelInference`` bound to the calibration graph.

        The calibration graph is the non-deployed ``GRAPH_FORWARD`` decoder
        graph; this selects it from the full per-graph maps and drives a
        ``DecoderInference`` (plus an ``EncoderInference``) from its metadata and
        export-input signature.

        Args:
            meta: ``{component: {graph_name: meta}}`` from :meth:`get_metadata`.
            example_inputs: ``{component: {graph_name: inputs}}`` from
                :meth:`get_example_inputs`.
            extra_options: Reads ``embedding_quantize`` to decide the token dtype.

        Returns:
            A ``ModelInference`` wrapping the calibration graph's
            ``DecoderInference`` plus an ``EncoderInference``, or ``None`` when no
            calibration graph is present.
        """
        from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
            ARTIFACT_TEXT_DECODER,
        )
        from executorch.backends.qualcomm.genai_pipeline.graph_names import (
            GRAPH_FORWARD,
        )
        from executorch.examples.qualcomm.oss_scripts.llama.inference import (
            DecoderInference,
            EncoderInference,
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
                audio_token_id=calibration_meta.get("audio_token_id"),
                image_token_id=calibration_meta.get("image_token_id"),
                max_context_len=calibration_meta["get_max_context_len"],
                max_batch_size=calibration_meta["get_max_batch_size"],
                use_i64_token=use_i64_token,
            ),
            encoder=EncoderInference(),
        )

    def load_tokenizer(
        self,
        model_name: str,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Build the model-level tokenizer wrapper."""
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

    def export_tokenizer(
        self,
        tokenizer: Any,
        output_dir: Path,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Return the runtime tokenizer file exported by ``TokenizerWrapper``."""
        runtime_tokenizer_path = Path(tokenizer.runtime_tokenizer_path)
        artifact_dir = Path(tokenizer.artifact)
        artifacts = list(artifact_dir.iterdir()) if artifact_dir.is_dir() else []
        if runtime_tokenizer_path not in artifacts:
            artifacts.append(runtime_tokenizer_path)
        return self._select_runtime_tokenizer(artifacts)

    def _build_custom_decoder_model(
        self,
        model_arch: Any,
    ) -> Dict[str, Any]:
        """Construct the decoder modules, without weights.

        ``model_arch`` is the decoder's ``{graph_name: constructor}`` map (already
        unwrapped from the component axis by :meth:`load_model`); each constructor
        is a ``partial`` with its config and every arch-specific kwarg pre-bound by
        ``get_model_arch``, so it is called with no arguments.
        """
        modules: Dict[str, Any] = {}
        for name, build in model_arch.items():
            modules[name] = build().eval()
        return modules

    def _load_decoder_state_dict(
        self,
        state_dict_loader: Optional[Callable[[str], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Acquire the decoder's raw checkpoint state dict.

        The checkpoint is whatever ``control_args`` names, or the model's
        ``repo_id`` downloaded and converted when it names none. Registry-specific
        HF loaders can be passed through ``model_options["state_dict_loader"]``
        and must accept only ``repo_id``.
        """
        import torch

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

            checkpoint = download_and_convert_hf_checkpoint(
                self.model_config.repo_id, self.model_config.convert_weights.__func__
            )

        return torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)

    def _load_auto_model(self) -> Any:
        """Load the shared HuggingFace model backing the encoders and embedding."""
        import torch
        from transformers import AutoModel, AutoModelForSpeechSeq2Seq

        repo_id = self.model_config.repo_id
        if ARTIFACT_AUDIO_ENCODER in self._modalities:
            auto_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                repo_id, _attn_implementation="eager"
            )
        else:
            auto_model = AutoModel.from_pretrained(
                repo_id, _attn_implementation="eager"
            )
        return auto_model.to(torch.float32).eval()

    @classmethod
    def from_model_config(
        cls,
        model_config: Any,
        *,
        control_args: Any,
    ) -> "MLLMLoaderAdapter":
        """Build the adapter.

        Args:
            model_config: The model's ``LLMModelConfig``.
            control_args: The CLI argument namespace.

        Returns:
            A new ``MLLMLoaderAdapter`` instance.
        """
        return cls(
            model_config=model_config,
            control_args=control_args,
        )
