# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging

import torch
import torchao
from packaging.version import parse
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

from executorch.extension.llm.optimum.integrations import ImageTextToTextExportableModule


# NOTE: It’s important to map the registered task name to the pipeline name in https://github.com/huggingface/transformers/blob/main/utils/update_metadata.py.
# This will streamline using inferred task names and make exporting models to Hugging Face pipelines easier.
def load_image_text_to_text_model(model_name_or_path: str, **kwargs):
    """
    Loads a causal language model for image-to-text generation and registers it under the task
    'image-text-to-text' using Hugging Face's AutoModelForCausalLM.

    Args:
        model_name_or_path (str):
            Model ID on huggingface.co or path on disk to the model repository to export. For example:
            `model_name_or_path="google/gemma-3-4b-it"` or `model_name_or_path="/path/to/model_folder`
        **kwargs:
            Additional configuration options for the model:
                - dtype (str, optional):
                    Data type for model weights (default: "float32").
                    Options include "float16" and "bfloat16".
                - attn_implementation (str, optional):
                    Attention mechanism implementation (default: "sdpa").
                - cache_implementation (str, optional):
                    Cache management strategy (default: "static").
                - max_length (int, optional):
                    Maximum sequence length for generation (default: 2048).

    Returns:
        ImageTextToTextExportableModule:
            An instance of `ImageTextToTextExportableModule` for exporting and lowering to ExecuTorch.
    """
    device = "cpu"
    batch_size = 1
    dtype = kwargs.get("dtype", "float32")
    use_custom_sdpa = kwargs.get("use_custom_sdpa", False)
    use_custom_kv_cache = kwargs.get("use_custom_kv_cache", False)
    attn_implementation = kwargs.get("attn_implementation", "custom_sdpa" if use_custom_sdpa else "sdpa")
    cache_implementation = kwargs.get("cache_implementation", "static")
    use_custom_sdpa = use_custom_sdpa or attn_implementation == "custom_sdpa"
    max_length = kwargs.get("max_length", 2048)
    config = kwargs.get("config") or AutoConfig.from_pretrained(model_name_or_path)

    # Make sure config has text_config and vision_config:
    if not hasattr(config, "text_config") or not hasattr(config, "vision_config"):
        raise ValueError(
            f"The model {model_name_or_path} does not have a `text_config` or `vision_config` attribute in its config. "
            "This is required for image-text-to-text models."
        )

    if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
        # NOTE: To make the model exportable we need to set the rope scaling to default to avoid hitting
        # the data-dependent control flow in _longrope_frequency_update. Alternatively, users should rewrite
        # that function to avoid the data-dependent control flow.
        config.rope_scaling["type"] = "default"

    if hasattr(config, "use_cache") and config.use_cache is False:
        config.use_cache = True

    eager_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device,
        torch_dtype=dtype,
        config=config,
        attn_implementation=attn_implementation,
        generation_config=GenerationConfig(
            use_cache=True,
            cache_implementation=cache_implementation,
            max_length=max_length,
            cache_config={
                "batch_size": batch_size,
                "max_cache_len": max_length,
            },
        ),
    )

    # Make sure model has language_model as well as vision_tower:
    if not hasattr(eager_model, "language_model") or not hasattr(eager_model, "vision_tower"):
        raise ValueError(
            f"The model {model_name_or_path} does not have a `language_model` or `vision_tower` attribute. "
            "This is required for image-text-to-text models."
        )

    for param in eager_model.parameters():
        # Must disable gradient for quantized checkpoint
        if isinstance(param, torchao.utils.TorchAOBaseTensor):
            param.requires_grad = False

    # TODO: Move quantization recipe out for better composability.
    # TODO: Should switch to `TorchAoConfig` once the quant issue on final lm_head layer is fixed.
    qlinear_config = kwargs.get("qlinear", None)
    qembedding_config = kwargs.get("qembedding", None)
    if qlinear_config or qembedding_config:
        # TODO: Update torchao to use 0.11.0 once released
        if parse(torchao.__version__) < parse("0.11.0.dev0"):
            raise RuntimeError("Quantization 8da4w requires torchao >= 0.11.0. Please upgrade torchao.")

        from torchao.quantization.granularity import PerAxis, PerGroup
        from torchao.quantization.quant_api import (
            Int8DynamicActivationIntxWeightConfig,
            IntxWeightOnlyConfig,
            quantize_,
        )
        from torchao.utils import unwrap_tensor_subclass

        if qembedding_config:
            logging.info("Quantizing embedding layers.")
            # TODO: Should switch to `AOPerModuleConfig` once fix for tied weights is available.
            embedding_config = IntxWeightOnlyConfig(
                weight_dtype=torch.int8,
                granularity=PerAxis(0),
            )
            quantize_(
                eager_model,
                embedding_config,
                lambda m, fqn: isinstance(m, torch.nn.Embedding),
            )

        if qlinear_config:
            logging.info("Quantizing linear layers.")
            linear_config = Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int4,
                weight_granularity=PerGroup(32),
            )
            quantize_(
                eager_model.language_model,
                linear_config,
            )

        unwrap_tensor_subclass(eager_model)

    return ImageTextToTextExportableModule(eager_model, use_custom_kv_cache, use_custom_sdpa)
