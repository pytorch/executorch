# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model lookup utilities for the GenAI Pipeline.

This module is the registry-facing entry point for model-specific pipeline
configuration. It keeps user-facing model names separate from the concrete
objects and options required by each pipeline stage.

Most callers should resolve a model config once with :func:`get_model_config`
and pass it to the more specific lookup helpers when available, so all derived
configuration comes from the same registry entry.
"""

from __future__ import annotations

import json
import logging
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_AUDIO_ENCODER,
    ARTIFACT_TEXT_DECODER,
    ARTIFACT_TOK_EMBEDDING,
    ARTIFACT_VISION_ENCODER,
)
from executorch.backends.qualcomm.genai_pipeline.graph_names import (
    DECODER_GRAPH_NAMES,
    GRAPH_FORWARD,
    TOK_EMBEDDING_GRAPH_NAMES,
)
from executorch.backends.qualcomm.genai_pipeline.model_components.decoder import (
    LlamaModel,
)
from executorch.backends.qualcomm.genai_pipeline.model_components.embedding import (
    TokenEmbedding,
)
from executorch.backends.qualcomm.genai_pipeline.models import LLM_VARIANT_ARCHS
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import ModelArgs
from executorch.examples.qualcomm.oss_scripts.llama.wrappers.base_component import (
    get_model_specific_kwargs,
    Mode,
    next_power_of_two,
    process_model_args,
)

logger = logging.getLogger(__name__)


def get_supported_models() -> List[str]:
    """Get list of all supported model names.

    Returns:
        Sorted list of supported model name strings.
    """
    from executorch.backends.qualcomm.genai_pipeline.models import SUPPORTED_LLM_MODELS

    return sorted(SUPPORTED_LLM_MODELS.keys())


def get_model_config(model_name: str) -> Any:
    """Look up a model configuration by name.
    Args:
        model_name: Model identifier (e.g., "llama3_2-1b_instruct", "qwen2_5-0_5b").
    Returns:
        The LLMModelConfig instance for the given model.
    Raises:
        KeyError: If model_name is not found in the registry.
    """
    from executorch.backends.qualcomm.genai_pipeline.models import SUPPORTED_LLM_MODELS

    model_name_lower = model_name.lower()
    if model_name_lower not in SUPPORTED_LLM_MODELS:
        available = get_supported_models()
        raise KeyError(
            f"Model '{model_name}' not found. " f"Available models: {available}"
        )

    config = SUPPORTED_LLM_MODELS[model_name_lower]

    return config


def get_model_arch_config(
    model_name: str, control_args: Any, model_config: Optional[Any] = None
) -> Any:
    """Resolve the decoder's config, for reading model shapes.

    All graphs (calibration / decode / prefill) share one set of weights and
    differ only in per-mode fields (``ar_len``, ``max_batch_size``,
    ``use_kv_cache``); the shapes source transforms read do not vary by graph.
    This returns a single config -- the decode graph's -- which
    :func:`get_source_transform` reads those shapes off.

    Args:
        model_name: The model's key in ``SUPPORTED_LLM_MODELS``.
        control_args: Carries paths and lengths.

    Returns:
        One decoder config (``ModelArgs``, or ``Gemma4Config`` for gemma4).

    Raises:
        ValueError: If the model needs a params file that ``control_args`` does
            not name.
    """
    if not model_config:
        model_config = get_model_config(model_name)

    if model_name == "gemma4-e2b":
        from executorch.examples.models.gemma4.text_decoder.gemma4_config import (
            Gemma4Config,
        )

        gemma4_config = Gemma4Config.from_json(model_config.params_path)
        gemma4_config.use_kv_cache = True
        gemma4_config.max_batch_size = 1
        gemma4_config.max_seq_len = control_args.max_seq_len
        gemma4_config.max_context_len = control_args.max_context_len
        return gemma4_config

    params_path = (
        model_config.params_path if control_args.params is None else control_args.params
    )
    if params_path is None:
        raise ValueError(
            f"Model '{model_name}' carries no params file, so its shapes "
            "cannot be resolved; pass one as control_args.params"
        )
    # The recipe is held as a CLASS, while ``process_model_args`` reads
    # ``get_kv_io_bit_width()`` off an instance.
    quant_recipe = model_config.quant_recipe
    if isinstance(quant_recipe, type):
        quant_recipe = quant_recipe()
    with open(params_path) as f:
        base_args = json.load(f)

    return process_model_args(
        control_args,
        ModelArgs(**base_args),
        quant_recipe,
        model_config,
        Mode.DECODE,
    )


def get_model_arch(
    model_name: str, control_args: Any, model_config: Optional[Any] = None
) -> Dict[str, Dict[str, Callable]]:
    """Build the model constructors exported for a model.

    Each constructor has its model config and graph-specific arguments
    pre-bound, allowing the loader to instantiate it without handling model
    modes or architecture details.

    The text decoder component is always included. The token embedding
    component is included only for multimodal models.

    Args:
        model_name: The model key in ``SUPPORTED_LLM_MODELS``.
        control_args: Runtime arguments controlling graph modes, shapes.
        model_config: Optional pre-resolved model configuration.

    Returns:
        A ``{component_name: {graph_name: constructor}}`` mapping.
    """

    # Get model config
    if not model_config:
        model_config = get_model_config(model_name)

    # Decode and calibration graphs are always exported. Prefill is additionally
    # exported for hybrid and lookahead modes.
    decode_graph_name, prefill_graph_name = DECODER_GRAPH_NAMES
    embedding_decode_name, embedding_prefill_name = TOK_EMBEDDING_GRAPH_NAMES
    graph_modes = [
        (decode_graph_name, embedding_decode_name, Mode.DECODE),
        (GRAPH_FORWARD, GRAPH_FORWARD, Mode.CALIBRATE),
        *(
            [(prefill_graph_name, embedding_prefill_name, Mode.PREFILL)]
            if control_args.model_mode in {"hybrid", "lookahead"}
            else []
        ),
    ]
    model_arch = {
        ARTIFACT_TEXT_DECODER: {},
        **({ARTIFACT_TOK_EMBEDDING: {}} if is_multimodal(model_name) else {}),
    }

    # Get any model-specific kwargs and quantization recipe to get the kv IO bit width.
    use_i64_token = control_args.embedding_quantize is not None
    model_specific_kwargs = get_model_specific_kwargs(control_args, model_config)
    quant_recipe = model_config.quant_recipe

    if model_name == "gemma4-e2b":
        from executorch.examples.models.gemma4.text_decoder.gemma4_config import (
            Gemma4Config,
        )
        from executorch.examples.qualcomm.oss_scripts.gemma4.model_wrapper import (
            Gemma4TextModelWrapper,
        )

        for decoder_graph_name, embedding_graph_name, mode in graph_modes:
            config = Gemma4Config.from_json(model_config.params_path)
            config.use_kv_cache = True
            config.max_batch_size = (
                control_args.batch_size if mode == Mode.CALIBRATE else 1
            )
            config.max_seq_len = control_args.max_seq_len
            config.max_context_len = control_args.max_context_len

            # Gemma 4 reads ar_len off the constructor, not the config.
            if mode == Mode.CALIBRATE:
                ar_len = control_args.max_context_len
            elif mode == Mode.PREFILL:
                ar_len = control_args.prefill_ar_len
            elif control_args.model_mode == "lookahead":
                ar_len = next_power_of_two(
                    (control_args.window + control_args.gcap) * (control_args.ngram - 1)
                )
            else:
                ar_len = 1

            extra_kwargs = {
                # 32 is the sentinel for "unquantized KV IO"; get_kv_io_bit_width()
                # returns it too when the recipe has no default_quant_dtype.
                "kv_io_bit_width": (
                    quant_recipe().get_kv_io_bit_width() if quant_recipe else 32
                ),
            }

            # Get Text Decoder model architecture.
            model_arch[ARTIFACT_TEXT_DECODER][decoder_graph_name] = partial(
                Gemma4TextModelWrapper,
                config,
                ar_len=ar_len,
                output_new_cache_only=True,
                output_cache=True,
                use_i64_token=use_i64_token,
                enable_masked_softmax=False,
                **extra_kwargs,
            )

            # Get Token Embedding model architecture if the model is multimodal LLM.
            if is_multimodal(model_name):
                model_arch[ARTIFACT_TOK_EMBEDDING][embedding_graph_name] = partial(
                    TokenEmbedding,
                    max_batch_size=config.max_batch_size,
                    ar_len=ar_len,
                    vocab_size=config.vocab_size,
                    dim=config.dim,
                    use_i64_token=use_i64_token,
                )
    else:
        params_path = (
            model_config.params_path
            if control_args.params is None
            else control_args.params
        )
        if params_path is None:
            raise ValueError(
                f"Model '{model_name}' carries no params file, so its shapes "
                "cannot be resolved; pass one as control_args.params"
            )

        with open(params_path) as f:
            base_args = json.load(f)

        for decoder_graph_name, embedding_graph_name, mode in graph_modes:
            config = process_model_args(
                control_args,
                ModelArgs(**base_args),
                quant_recipe(mode == Mode.CALIBRATE),
                model_config,
                mode,
            )
            # TODO: Decouple example-input generation from the model because
            # input shapes are graph-specific rather than model-specific.
            model_arch[ARTIFACT_TEXT_DECODER][decoder_graph_name] = partial(
                LLM_VARIANT_ARCHS.get(model_name, LlamaModel),
                config,
                ar_len=config.ar_len,
                output_new_cache_only=True,
                output_cache=True,
                use_i64_token=use_i64_token,
                **model_specific_kwargs,
            )

            # Get Token Embedding model architecture if the model is multimodal LLM.
            if is_multimodal(model_name):
                model_arch[ARTIFACT_TOK_EMBEDDING][embedding_graph_name] = partial(
                    TokenEmbedding,
                    max_batch_size=config.max_batch_size,
                    ar_len=config.ar_len,
                    vocab_size=config.vocab_size,
                    dim=config.dim,
                    use_i64_token=use_i64_token,
                )

    return model_arch


def get_source_transform(
    model_name: str,
    *,
    control_args: Any,
    model_config: Optional[Any] = None,
) -> Tuple[Dict[str, List[Callable]], Dict[str, List[Callable]]]:
    """Resolve a model's source transforms, mirroring ``LLMWrapper._prepare_model``.

    Args:
        model_name: The model's key in ``SUPPORTED_LLM_MODELS``.
        control_args: CLI arguments used to determine the checkpoint source and
            resolve the values bound to individual transforms.
        model_config: Optional resolved model configuration. When omitted, it
            is looked up from ``model_name``.

    Returns:
        A ``(weight_transforms, module_transforms)`` tuple of component-keyed
        transform maps:

        - ``weight_transforms``: Each callable accepts a state dict and returns
          the transformed state dict before model weights are loaded.
        - ``module_transforms``: Each callable accepts a module and returns the
          transformed module after model weights are loaded.

        Components without transforms are omitted from each map.

    Raises:
        RuntimeError: If the model config requests SpinQuant (``r1``/``r2``),
            which is no longer supported -- matching the reference's guard.
    """
    from executorch.backends.qualcomm.genai_pipeline.source_transform import (
        apply_dtype_override,
        convert_linear_to_conv2d,
        gemma_rmsnorm_offset,
        permute_partial_rope,
        prepare_conv_submodules,
        remap_gemma4_keys,
        scale_token_embedding,
        strip_orig_mod_prefix,
        unwrap_model_key,
    )

    if model_config is None:
        model_config = get_model_config(model_name)
    config = model_config
    name = model_name.lower()
    is_hf_path = control_args.checkpoint is None
    model_args = get_model_arch_config(model_name, control_args, config)

    if config.r1 or config.r2:
        raise RuntimeError(
            "SpinQuant (r1/r2) is no longer supported: the "
            "torchao.prototype.spinquant module has been deleted."
        )

    # Weight chain, in ``_prepare_model`` order.
    weight_transforms = [unwrap_model_key]
    if is_hf_path:
        # HF path: gemma4 key rename, then the Gemma RMSNorm +1 offset, then the
        # embedding scale (self-guarding on the factor / key).
        if name == "gemma4-e2b":
            weight_transforms.append(remap_gemma4_keys)
        else:
            if name in ("gemma-2b", "gemma2-2b", "gemma3-1b"):
                weight_transforms.append(gemma_rmsnorm_offset)
            weight_transforms.append(
                partial(
                    scale_token_embedding,
                    embedding_scale_factor=model_args.embedding_scale_factor,
                )
            )
    else:
        # Local checkpoint path: stories260k carries torch.compile's
        # ``_orig_mod.`` prefix and must be renamed.
        if name == "stories260k":
            weight_transforms.append(strip_orig_mod_prefix)
    # RoPE weight layout permutation, gated on the model config flag.
    if config.transform_weight:
        weight_transforms.append(
            partial(
                permute_partial_rope,
                n_layers=model_args.n_layers,
                n_heads=model_args.n_heads,
                n_kv_heads=model_args.n_kv_heads,
                partial_rotary_factor=model_args.partial_rotary_factor,
            )
        )

    # Module chain, in ``_prepare_model`` order: submodule conv prep, the
    # linear-to-conv2d rewrite it feeds, then the dtype override last so it also
    # covers the conv2d modules.
    module_transforms = [
        prepare_conv_submodules,
        convert_linear_to_conv2d,
        partial(
            apply_dtype_override,
            dtype_override=control_args.dtype_override,
        ),
    ]

    return {ARTIFACT_TEXT_DECODER: weight_transforms}, {
        ARTIFACT_TEXT_DECODER: module_transforms
    }


def load_hf_checkpoint_state_dict(
    repo_id: str,
    *,
    convert_weights: Callable,
) -> Dict[str, Any]:
    """Download a standard HF checkpoint and return its converted state dict."""
    import torch
    from executorch.examples.models.llama.hf_download import (
        download_and_convert_hf_checkpoint,
    )

    checkpoint = download_and_convert_hf_checkpoint(repo_id, convert_weights)
    return torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)


def load_gemma4_hf_checkpoint_state_dict(
    repo_id: str,
    *,
    convert_weights: Callable,
    gemma4_config: Any,
    dtype: Any,
) -> Dict[str, Any]:
    """Download Gemma4 HF weights and return the converter-produced state dict."""
    from huggingface_hub import snapshot_download

    return convert_weights(
        snapshot_download(repo_id=repo_id),
        gemma4_config,
        dtype,
    )


def get_state_dict_loader(
    model_name: str,
    *,
    control_args: Any,
    model_config: Optional[Any] = None,
) -> Dict[str, Callable[[str], Dict[str, Any]]]:
    """Return component-keyed loaders for remote Hugging Face checkpoints.

    Args:
        model_name: Registered model identifier used to select the loader.
        control_args: Runtime arguments. A configured local ``checkpoint``
            bypasses remote loading and returns no loader.
        model_config: Optional resolved model configuration. When omitted, it
            is looked up from ``model_name``.

    Returns:
        A component-keyed loader map. Each loader accepts a Hugging Face
        ``repo_id`` and returns a converted state dict:

        - When ``control_args.checkpoint`` is set: An empty map, because local
          checkpoint loading does not need a remote loader.
        - Otherwise: A map containing the text-decoder loader. The loader
          has model-specific conversion details bound into the callable.

    Raises:
        KeyError: If ``model_name`` is not registered and ``model_config`` is
            not supplied.
    """
    import torch

    if model_config is None:
        model_config = get_model_config(model_name)
    if control_args.checkpoint is not None:
        return {}

    convert_weights = model_config.convert_weights
    if hasattr(convert_weights, "__func__"):
        convert_weights = convert_weights.__func__

    name = model_name.lower()
    if name == "gemma4-e2b":
        return {
            ARTIFACT_TEXT_DECODER: partial(
                load_gemma4_hf_checkpoint_state_dict,
                convert_weights=convert_weights,
                gemma4_config=get_model_arch_config(
                    model_name, control_args, model_config
                ),
                dtype=torch.float32,
            )
        }

    return {
        ARTIFACT_TEXT_DECODER: partial(
            load_hf_checkpoint_state_dict,
            convert_weights=convert_weights,
        )
    }


def get_model_num_sharding(model_name: str) -> Dict[str, int]:
    """Get the number of graph shards for each component of a model.

    Args:
        model_name: Model identifier.

    Returns:
        ``{component_name: num_shards}`` (1 = no sharding). The text decoder
        is always present; vision/audio encoders are present only if the
        model declares them.

    Raises:
        KeyError: If model_name is not found.
    """
    config = get_model_config(model_name)
    num_sharding = {ARTIFACT_TEXT_DECODER: getattr(config, "num_sharding", 1)}
    if hasattr(config, "vision_encoder"):
        num_sharding[ARTIFACT_VISION_ENCODER] = config.vision_encoder.num_sharding
    if hasattr(config, "audio_encoder"):
        num_sharding[ARTIFACT_AUDIO_ENCODER] = config.audio_encoder.num_sharding
    return num_sharding


def get_quant_dtype(model_name: str) -> Any:
    """Get component-keyed quantization dtypes for the model.

    Components without an explicit model configuration default to the higher
    precision ``QuantDtype.use_16a8w`` (16A8W).

    Returns:
        Component-keyed quantization dtypes.

    Raises:
        KeyError: If model_name is not found.
    """
    from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype

    config = get_model_config(model_name)
    quant_dtype = {
        ARTIFACT_TEXT_DECODER: getattr(config, "quant_dtype", QuantDtype.use_16a8w)
    }
    if hasattr(config, "vision_encoder"):
        quant_dtype[ARTIFACT_TOK_EMBEDDING] = QuantDtype.use_16a8w
        quant_dtype[ARTIFACT_VISION_ENCODER] = getattr(
            config.vision_encoder, "quant_dtype", QuantDtype.use_16a8w
        )
    if hasattr(config, "audio_encoder"):
        quant_dtype[ARTIFACT_TOK_EMBEDDING] = QuantDtype.use_16a8w
        quant_dtype[ARTIFACT_AUDIO_ENCODER] = getattr(
            config.audio_encoder, "quant_dtype", QuantDtype.use_16a8w
        )

    return quant_dtype


def get_quant_recipe(model_name: str) -> Any:
    """Get the quantization recipe class for the model.

    Returns:
        Component-keyed quantization recipe classes.
    """

    config = get_model_config(model_name)

    quant_recipe = {ARTIFACT_TEXT_DECODER: getattr(config, "quant_recipe", None)}
    if hasattr(config, "vision_encoder"):
        quant_recipe[ARTIFACT_TOK_EMBEDDING] = None
        quant_recipe[ARTIFACT_VISION_ENCODER] = getattr(
            config.vision_encoder, "quant_recipe", None
        )
    if hasattr(config, "audio_encoder"):
        quant_recipe[ARTIFACT_TOK_EMBEDDING] = None
        quant_recipe[ARTIFACT_AUDIO_ENCODER] = getattr(
            config.audio_encoder, "quant_recipe", None
        )
    return quant_recipe


def is_multimodal(model_name: str) -> bool:
    """Check if a model is multimodal (has vision/audio encoders).

    Args:
        model_name: Model identifier.

    Returns:
        True if the model has vision or audio encoder capabilities.

    Raises:
        KeyError: If model_name is not found.
    """
    config = get_model_config(model_name)
    return hasattr(config, "audio_encoder") or hasattr(config, "vision_encoder")


def get_model_loader_adapter(
    model_name: str,
    control_args: Any,
) -> Any:
    """Get the model loader adapter for ``model_name``.

    The returned adapter is selected from the model registry: multimodal models
    use ``MLLMLoaderAdapter`` and text-only models use ``LLMLoaderAdapter``.
    Model architecture and source transforms are resolved separately and passed
    through ``ModelPreparationInputConfig.extra_options``.

    Args:
        model_name: Model identifier (e.g., "llama3_2-1b_instruct").
        control_args: CLI namespace carrying paths and shape controls.

    Returns:
        The loader adapter for the model.

    Raises:
        KeyError: If model_name is not found in the registry.
    """
    config = get_model_config(model_name)

    if is_multimodal(model_name):
        from executorch.backends.qualcomm.genai_pipeline.strategies.model_preparation.mllm_loader_adapter import (
            MLLMLoaderAdapter,
        )

        adapter_cls = MLLMLoaderAdapter
        logger.debug("Created multimodal model loader adapter for '%s'", model_name)
    else:
        from executorch.backends.qualcomm.genai_pipeline.strategies.model_preparation.llm_loader_adapter import (
            LLMLoaderAdapter,
        )

        adapter_cls = LLMLoaderAdapter
        logger.debug("Created static LLM model loader adapter for '%s'", model_name)

    return adapter_cls.from_model_config(
        config,
        control_args=control_args,
    )


def get_quantizer_adapter(model_name: str) -> Any:
    """Create a quantizer adapter for the given model.

    For LLM models, returns LLMQuantizerAdapter.
    For multimodal models (VLM/ALM), returns MLLMQuantizerAdapter.

    Recipes and dtypes are routed by ``ExecuTorchQuantizationStrategy`` from the
    component-keyed ``quantize_options`` dict.

    Args:
        model_name: Model identifier (e.g., "llama3_2-1b_instruct", "smolvlm_500m").

    Returns:
        LLMQuantizerAdapter for LLM models, MLLMQuantizerAdapter for multimodal models.

    Raises:
        KeyError: If model_name is not found in the registry.
    """
    if is_multimodal(model_name):
        from executorch.backends.qualcomm.genai_pipeline.strategies.quantization.mllm_quantizer_adapter import (
            MLLMQuantizerAdapter,
        )

        adapter = MLLMQuantizerAdapter()
        logger.debug("Created MLLMQuantizerAdapter for '%s'", model_name)
        return adapter
    else:
        from executorch.backends.qualcomm.genai_pipeline.strategies.quantization.llm_quantizer_adapter import (
            LLMQuantizerAdapter,
        )

        adapter = LLMQuantizerAdapter()

        logger.debug("Created LLMQuantizerAdapter for '%s'", model_name)
        return adapter
